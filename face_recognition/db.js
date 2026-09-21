const fs = require('fs');
const path = require('path');
const { DatabaseSync } = require('node:sqlite');

const DATA_DIR = path.join(__dirname, 'data');
const SQLITE_FILE = path.join(DATA_DIR, 'face_recognition.db');
const UPLOADS_DIR = path.join(DATA_DIR, 'uploads');
const CROPS_DIR = path.join(DATA_DIR, 'crops');
const LEGACY_JSON_FILE = path.join(DATA_DIR, 'database.json');

if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
if (!fs.existsSync(UPLOADS_DIR)) fs.mkdirSync(UPLOADS_DIR, { recursive: true });
if (!fs.existsSync(CROPS_DIR)) fs.mkdirSync(CROPS_DIR, { recursive: true });

const sqlite = new DatabaseSync(SQLITE_FILE, {
  enableForeignKeyConstraints: true,
  timeout: 5000
});
sqlite.exec(`
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS persons (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  gender TEXT NOT NULL DEFAULT 'Unknown',
  created_at TEXT
);
CREATE TABLE IF NOT EXISTS photos (
  id TEXT PRIMARY KEY,
  person_id TEXT NOT NULL,
  filename TEXT,
  embedding TEXT,
  FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY,
  timestamp TEXT,
  person_id TEXT,
  person_name TEXT,
  score REAL,
  crop_filename TEXT,
  is_known INTEGER NOT NULL DEFAULT 0,
  camera_id TEXT,
  camera_name TEXT
);
CREATE TABLE IF NOT EXISTS cameras (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  rtsp_url TEXT,
  is_active INTEGER NOT NULL DEFAULT 0,
  line_crossing_enabled INTEGER NOT NULL DEFAULT 0,
  line_y REAL NOT NULL DEFAULT 0.6,
  line_direction TEXT NOT NULL DEFAULT 'in',
  line_x_start REAL NOT NULL DEFAULT 0,
  line_x_end REAL NOT NULL DEFAULT 1,
  created_at TEXT
);
CREATE TABLE IF NOT EXISTS clusters (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  gender TEXT NOT NULL DEFAULT 'Unknown',
  created_at TEXT
);
CREATE TABLE IF NOT EXISTS cluster_photos (
  row_id INTEGER PRIMARY KEY AUTOINCREMENT,
  cluster_id TEXT NOT NULL,
  photo_id TEXT,
  filename TEXT,
  embedding TEXT,
  gender TEXT NOT NULL DEFAULT 'Unknown',
  FOREIGN KEY (cluster_id) REFERENCES clusters(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_photos_person ON photos(person_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_events_person ON events(person_id);
CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera_id);
CREATE INDEX IF NOT EXISTS idx_cluster_photos_cluster ON cluster_photos(cluster_id);
`);

function parseEmbedding(value) {
  if (value === null || value === undefined || value === '') return null;
  try { return JSON.parse(value); } catch (_) { return null; }
}

function emptyDB() {
  return { persons: [], photos: [], events: [], cameras: [], clusters: [], cluster_counter: 0, settings: {} };
}

function readDB() {
  const data = emptyDB();

  data.persons = sqlite.prepare(
    'SELECT id,name,gender,created_at FROM persons ORDER BY rowid'
  ).all();

  data.photos = sqlite.prepare(
    'SELECT id,person_id,filename,embedding FROM photos ORDER BY rowid'
  ).all().map(p => ({ ...p, embedding: parseEmbedding(p.embedding) }));

  data.events = sqlite.prepare(
    'SELECT id,timestamp,person_id,person_name,score,crop_filename,is_known,camera_id,camera_name FROM events ORDER BY rowid'
  ).all().map(e => ({ ...e, is_known: !!e.is_known }));

  data.cameras = sqlite.prepare(
    'SELECT id,name,rtsp_url,is_active,line_crossing_enabled,line_y,line_direction,line_x_start,line_x_end,created_at FROM cameras ORDER BY rowid'
  ).all().map(c => ({
    ...c,
    is_active: !!c.is_active,
    line_crossing_enabled: !!c.line_crossing_enabled
  }));

  const clusterRows = sqlite.prepare(
    'SELECT id,name,gender,created_at FROM clusters ORDER BY rowid'
  ).all();
  const clusterPhotos = sqlite.prepare(
    'SELECT row_id,cluster_id,photo_id,filename,embedding,gender FROM cluster_photos ORDER BY row_id'
  ).all();

  data.clusters = clusterRows.map(c => ({
    ...c,
    photos: clusterPhotos
      .filter(p => p.cluster_id === c.id)
      .map(p => ({
        ...(p.photo_id ? { id: p.photo_id } : {}),
        filename: p.filename,
        embedding: parseEmbedding(p.embedding),
        gender: p.gender || 'Unknown'
      }))
  }));

  const counter = sqlite.prepare("SELECT value FROM meta WHERE key='cluster_counter'").get();
  data.cluster_counter = counter ? Number(counter.value) || 0 : data.clusters.length;

  for (const row of sqlite.prepare('SELECT key,value FROM settings').all()) {
    try { data.settings[row.key] = JSON.parse(row.value); }
    catch (_) { data.settings[row.key] = row.value; }
  }

  return data;
}

function writeDB(input) {
  sqlite.exec('BEGIN IMMEDIATE');
  try {
  const data = { ...emptyDB(), ...(input || {}) };

  sqlite.exec(`
    DELETE FROM cluster_photos;
    DELETE FROM clusters;
    DELETE FROM photos;
    DELETE FROM persons;
    DELETE FROM events;
    DELETE FROM cameras;
    DELETE FROM settings;
    DELETE FROM meta;
  `);

  const personStmt = sqlite.prepare(
    'INSERT INTO persons(id,name,gender,created_at) VALUES(?,?,?,?)'
  );
  for (const p of (data.persons || [])) {
    personStmt.run(p.id, p.name || '', p.gender || 'Unknown', p.created_at || null);
  }

  const photoStmt = sqlite.prepare(
    'INSERT INTO photos(id,person_id,filename,embedding) VALUES(?,?,?,?)'
  );
  for (const p of (data.photos || [])) {
    photoStmt.run(
      p.id, p.person_id || null, p.filename || null,
      p.embedding == null ? null : JSON.stringify(p.embedding)
    );
  }

  const eventStmt = sqlite.prepare(
    'INSERT INTO events(id,timestamp,person_id,person_name,score,crop_filename,is_known,camera_id,camera_name) VALUES(?,?,?,?,?,?,?,?,?)'
  );
  for (const e of (data.events || [])) {
    eventStmt.run(
      e.id, e.timestamp || null, e.person_id || null, e.person_name || null,
      Number(e.score || 0), e.crop_filename || null, e.is_known ? 1 : 0,
      e.camera_id || null, e.camera_name || null
    );
  }

  const cameraStmt = sqlite.prepare(
    'INSERT INTO cameras(id,name,rtsp_url,is_active,line_crossing_enabled,line_y,line_direction,line_x_start,line_x_end,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)'
  );
  for (const c of (data.cameras || [])) {
    cameraStmt.run(
      c.id, c.name || '', c.rtsp_url || '', c.is_active ? 1 : 0,
      c.line_crossing_enabled ? 1 : 0, Number(c.line_y ?? 0.6),
      c.line_direction || 'in', Number(c.line_x_start ?? 0),
      Number(c.line_x_end ?? 1), c.created_at || null
    );
  }

  const clusterStmt = sqlite.prepare(
    'INSERT INTO clusters(id,name,gender,created_at) VALUES(?,?,?,?)'
  );
  const clusterPhotoStmt = sqlite.prepare(
    'INSERT INTO cluster_photos(cluster_id,photo_id,filename,embedding,gender) VALUES(?,?,?,?,?)'
  );
  for (const c of (data.clusters || [])) {
    clusterStmt.run(c.id, c.name || '', c.gender || 'Unknown', c.created_at || null);
    for (const p of (c.photos || [])) {
      clusterPhotoStmt.run(
        c.id, p.id || null, p.filename || null,
        p.embedding == null ? null : JSON.stringify(p.embedding),
        p.gender || 'Unknown'
      );
    }
  }

  const settingStmt = sqlite.prepare('INSERT INTO settings(key,value) VALUES(?,?)');
  for (const [key, value] of Object.entries(data.settings || {})) {
    settingStmt.run(key, JSON.stringify(value));
  }

  sqlite.prepare('INSERT INTO meta(key,value) VALUES(?,?)')
    .run('cluster_counter', String(Number(data.cluster_counter) || 0));
    sqlite.exec('COMMIT');
  } catch (err) {
    try { sqlite.exec('ROLLBACK'); } catch (_) {}
    throw err;
  }
}


function migrateLegacyJSON() {
  const hasRows = sqlite.prepare('SELECT EXISTS(SELECT 1 FROM persons) AS x').get().x ||
                  sqlite.prepare('SELECT EXISTS(SELECT 1 FROM photos) AS x').get().x ||
                  sqlite.prepare('SELECT EXISTS(SELECT 1 FROM events) AS x').get().x ||
                  sqlite.prepare('SELECT EXISTS(SELECT 1 FROM cameras) AS x').get().x ||
                  sqlite.prepare('SELECT EXISTS(SELECT 1 FROM clusters) AS x').get().x;
  if (hasRows || !fs.existsSync(LEGACY_JSON_FILE)) return;

  try {
    const legacy = JSON.parse(fs.readFileSync(LEGACY_JSON_FILE, 'utf8'));
    writeDB({ ...emptyDB(), ...legacy });
    const backup = LEGACY_JSON_FILE + '.migrated';
    fs.renameSync(LEGACY_JSON_FILE, backup);
    console.log('[SQLite] Migrated legacy database.json to face_recognition.db.');
    console.log('[SQLite] Legacy JSON renamed to database.json.migrated.');
  } catch (err) {
    console.error('[SQLite] Legacy JSON migration failed:', err.message);
    throw err;
  }
}

migrateLegacyJSON();

const api = {
  getSettings() {
    return readDB().settings || {};
  },
  setSetting(key, value) {
    const stmt = sqlite.prepare(
      'INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value'
    );
    stmt.run(key, JSON.stringify(value));
  }
};

// Math helpers for vector comparisons with L2 normalization
function dotProduct(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function l2Distance(a, b) {
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);
  
  if (normA === 0 || normB === 0) return 2.0; // Max possible L2 distance of normalized vectors
  
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = (a[i] / normA) - (b[i] / normB);
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

module.exports = {
  DATA_DIR,
  SQLITE_FILE,
  getSettings() {
    return readDB().settings || {};
  },
  setSetting(key, value) {
    sqlite.prepare('INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value')
      .run(key, JSON.stringify(value));
  },
  UPLOADS_DIR,
  CROPS_DIR,

  // Persons
  getPersons() {
    const db = readDB();
    return db.persons.map(p => {
      const pPhotos = db.photos.filter(ph => ph.person_id === p.id).map(ph => ({
        id: ph.id,
        filename: ph.filename
      }));
      return { ...p, photos: pPhotos };
    });
  },

  getPerson(id) {
    const db = readDB();
    const p = db.persons.find(p => p.id === id);
    if (!p) return null;
    const pPhotos = db.photos.filter(ph => ph.person_id === p.id).map(ph => ({
      id: ph.id,
      filename: ph.filename
    }));
    return { ...p, photos: pPhotos };
  },

  addPerson(name, gender = 'Unknown') {
    const db = readDB();
    const newPerson = {
      id: Date.now().toString(36) + Math.random().toString(36).substr(2, 5),
      name,
      gender: gender || 'Unknown',
      created_at: new Date().toISOString()
    };
    db.persons.push(newPerson);
    writeDB(db);
    return newPerson;
  },

  deletePerson(id) {
    const db = readDB();
    const photosToDelete = db.photos.filter(p => p.person_id === id);
    photosToDelete.forEach(p => {
      const filePath = path.join(UPLOADS_DIR, p.filename);
      if (fs.existsSync(filePath)) {
        try { fs.unlinkSync(filePath); } catch (e) {}
      }
    });

    db.persons = db.persons.filter(p => p.id !== id);
    db.photos = db.photos.filter(p => p.person_id !== id);
    writeDB(db);
    return true;
  },

  // Photos
  addPhoto(personId, filename, embedding) {
    const db = readDB();
    const person = db.persons.find(p => p.id === personId);
    if (!person) throw new Error('Person not found');

    const newPhoto = {
      id: Date.now().toString(36) + Math.random().toString(36).substr(2, 5),
      person_id: personId,
      filename,
      embedding
    };
    db.photos.push(newPhoto);
    writeDB(db);
    return newPhoto;
  },

  deletePhoto(photoId) {
    const db = readDB();
    const photo = db.photos.find(p => p.id === photoId);
    if (!photo) return false;

    const filePath = path.join(UPLOADS_DIR, photo.filename);
    if (fs.existsSync(filePath)) {
      try { fs.unlinkSync(filePath); } catch (e) {}
    }

    db.photos = db.photos.filter(p => p.id !== photoId);
    writeDB(db);
    return true;
  },

  getAllCandidates() {
    const db = readDB();
    return db.persons.map(p => {
      const embeddings = db.photos
        .filter(ph => ph.person_id === p.id)
        .map(ph => ph.embedding);
      return {
        person_id: p.id,
        name: p.name,
        embeddings: embeddings
      };
    }).filter(c => c.embeddings.length > 0);
  },

  // Cameras
  getCameras() {
    const db = readDB();
    return db.cameras || [];
  },

  getCamera(id) {
    const db = readDB();
    return db.cameras.find(c => c.id === id) || null;
  },

  updateCamera(id, updates = {}) {
    const db = readDB();
    const camera = db.cameras.find(c => c.id === id);
    if (!camera) return null;
    Object.assign(camera, updates);
    writeDB(db);
    return camera;
  },

  addCamera(name, rtspUrl) {
    const db = readDB();
    const newCamera = {
      id: 'cam_' + Date.now().toString(36) + Math.random().toString(36).substr(2, 4),
      name,
      rtsp_url: rtspUrl,
      is_active: false,
      line_crossing_enabled: false,
      line_y: 0.6,
      line_direction: 'in',
      line_x_start: 0.0,
      line_x_end: 1.0,
      created_at: new Date().toISOString()
    };
    db.cameras.push(newCamera);
    writeDB(db);
    return newCamera;
  },

  deleteCamera(id) {
    const db = readDB();
    db.cameras = db.cameras.filter(c => c.id !== id);
    writeDB(db);
    return true;
  },

  updateCameraStatus(id, isActive) {
    const db = readDB();
    const camera = db.cameras.find(c => c.id === id);
    if (camera) {
      camera.is_active = !!isActive;
      writeDB(db);
      return camera;
    }
    return null;
  },

  updateCameraLineSettings(id, enabled, lineY, direction, lineXStart, lineXEnd) {
    const db = readDB();
    const camera = db.cameras.find(c => c.id === id);
    if (camera) {
      camera.line_crossing_enabled = !!enabled;
      if (lineY !== undefined) camera.line_y = parseFloat(lineY);
      if (direction !== undefined) camera.line_direction = direction;
      if (lineXStart !== undefined) camera.line_x_start = parseFloat(lineXStart);
      if (lineXEnd !== undefined) camera.line_x_end = parseFloat(lineXEnd);
      writeDB(db);
      return camera;
    }
    return null;
  },

  // Events
  getEvents(limit = 100) {
    const db = readDB();
    return db.events
      .slice()
      .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
      .slice(0, limit);
  },

  addEvent(personId, personName, score, cropFilename, isKnown, cameraId = null, cameraName = 'Manual Upload') {
    const db = readDB();
    const newEvent = {
      id: Date.now().toString(36) + Math.random().toString(36).substr(2, 5),
      timestamp: new Date().toISOString(),
      person_id: personId,
      person_name: personName,
      score: score,
      crop_filename: cropFilename,
      is_known: isKnown,
      camera_id: cameraId,
      camera_name: cameraName
    };
    db.events.push(newEvent);
    
    // Capping at 1000 events
    if (db.events.length > 1000) {
      const removed = db.events.shift();
      if (removed && removed.crop_filename) {
        const cropPath = path.join(CROPS_DIR, removed.crop_filename);
        if (fs.existsSync(cropPath)) {
          try { fs.unlinkSync(cropPath); } catch (e) {}
        }
      }
    }
    writeDB(db);
    return newEvent;
  },

  updateEvent(eventId, updates) {
    const db = readDB();
    const eventIndex = db.events.findIndex(e => e.id === eventId);
    if (eventIndex !== -1) {
      db.events[eventIndex] = { ...db.events[eventIndex], ...updates };
      writeDB(db);
      return db.events[eventIndex];
    }
    return null;
  },

  clearEvents() {
    const db = readDB();
    db.events.forEach(ev => {
      if (ev.crop_filename) {
        const cropPath = path.join(CROPS_DIR, ev.crop_filename);
        if (fs.existsSync(cropPath)) {
          try { fs.unlinkSync(cropPath); } catch (e) {}
        }
      }
    });
    db.events = [];
    writeDB(db);
    return true;
  },

  // Clusters
  getClusters() {
    const db = readDB();
    return db.clusters || [];
  },

  addFaceToCluster(embedding, cropFilename, gender, settings) {
    const db = readDB();
    if (!db.clusters) db.clusters = [];
    if (typeof db.cluster_counter !== 'number') {
      db.cluster_counter = db.clusters.length;
    }
    
    const disType = settings.dis_type; // 0 cosine, 1 norml2
    
    // Apply a threshold for clustering to allow grouping of similar faces while separating different people
    let threshold;
    if (disType === 0) { // Cosine
      threshold = Math.max(0.58, settings.threshold + 0.02);
    } else { // L2
      threshold = Math.min(0.95, settings.threshold - 0.15);
    }
    
    let bestCluster = null;
    let bestScore = disType === 0 ? -1.0 : Infinity;
    
    for (const cluster of db.clusters) {
      let clusterScores = [];
      for (const photo of cluster.photos) {
        let score;
        if (disType === 0) { // Cosine
          score = dotProduct(embedding, photo.embedding);
        } else { // L2
          score = l2Distance(embedding, photo.embedding);
        }
        clusterScores.push(score);
      }
      
      const avgScore = clusterScores.reduce((sum, s) => sum + s, 0) / clusterScores.length;
      
      if (disType === 0) { // Cosine: higher score is better
        if (avgScore >= threshold && avgScore > bestScore) {
          bestScore = avgScore;
          bestCluster = cluster;
        }
      } else { // L2: lower score is better
        if (avgScore <= threshold && avgScore < bestScore) {
          bestScore = avgScore;
          bestCluster = cluster;
        }
      }
    }
    
    if (bestCluster) {
      bestCluster.photos.push({
        filename: cropFilename,
        embedding: embedding,
        gender: gender || 'Unknown'
      });
      
      // Calculate majority vote gender of cluster
      const votes = { 'Male': 0, 'Female': 0 };
      bestCluster.photos.forEach(ph => {
        if (ph.gender === 'Male' || ph.gender === 'Female') {
          votes[ph.gender]++;
        }
      });
      
      let finalGender = 'Unknown';
      if (votes['Male'] > votes['Female']) {
        finalGender = 'Male';
      } else if (votes['Female'] > votes['Male']) {
        finalGender = 'Female';
      }
      
      bestCluster.gender = finalGender;
      const genderSuffix = finalGender !== 'Unknown' ? ` (${finalGender})` : '';
      bestCluster.name = bestCluster.name.split(' (')[0] + genderSuffix;
      
      writeDB(db);
      return { id: bestCluster.id, name: bestCluster.name, gender: bestCluster.gender, isNew: false };
    } else {
      // Use persistent auto-incrementing counter
      db.cluster_counter = (db.cluster_counter || 0) + 1;
      const nextNum = db.cluster_counter;
      const genderSuffix = (gender && gender !== 'Unknown') ? ` (${gender})` : '';
      const clusterId = 'cluster_' + Date.now().toString(36) + Math.random().toString(36).substr(2, 4);
      const newCluster = {
        id: clusterId,
        name: `Profile #${nextNum}${genderSuffix}`,
        gender: gender || 'Unknown',
        photos: [{
          filename: cropFilename,
          embedding: embedding,
          gender: gender || 'Unknown'
        }],
        created_at: new Date().toISOString()
      };
      db.clusters.push(newCluster);
      writeDB(db);
      return { id: clusterId, name: newCluster.name, gender: newCluster.gender, isNew: true };
    }
  },

  enrollCluster(clusterId, name) {
    const db = readDB();
    if (!db.clusters) db.clusters = [];
    
    const cluster = db.clusters.find(c => c.id === clusterId);
    if (!cluster) throw new Error('Cluster not found');
    
    // Check if person with same name already exists (case-insensitive) to merge them
    let person = db.persons.find(p => p.name.toLowerCase() === name.trim().toLowerCase());
    const isNew = !person;
    if (isNew) {
      person = {
        id: Date.now().toString(36) + Math.random().toString(36).substr(2, 5),
        name: name.trim(),
        gender: cluster.gender || 'Unknown',
        created_at: new Date().toISOString()
      };
      db.persons.push(person);
    } else {
      // Merge: Update gender if existing was Unknown and cluster is not
      if (cluster.gender && cluster.gender !== 'Unknown' && (!person.gender || person.gender === 'Unknown')) {
        person.gender = cluster.gender;
      }
    }
    
    cluster.photos.forEach(ph => {
      const cropPath = path.join(CROPS_DIR, ph.filename);
      const uploadFilename = `enrolled_${Date.now()}_${ph.filename}`;
      const uploadPath = path.join(UPLOADS_DIR, uploadFilename);
      
      try {
        if (fs.existsSync(cropPath)) {
          fs.copyFileSync(cropPath, uploadPath);
        }
      } catch (err) {
        console.error('Failed to copy crop file to uploads during enrollment:', err);
      }
      
      db.photos.push({
        id: Date.now().toString(36) + Math.random().toString(36).substr(2, 4),
        person_id: person.id,
        filename: uploadFilename,
        embedding: ph.embedding
      });
    });
    
    db.events.forEach(ev => {
      if (ev.person_id === clusterId) {
        ev.person_id = person.id;
        ev.person_name = isNew ? person.name : `${person.name} (${person.gender && person.gender !== 'Unknown' ? person.gender : 'Unknown'})`;
        ev.is_known = true;
      }
    });
    
    db.clusters = db.clusters.filter(c => c.id !== clusterId);
    writeDB(db);
    return person;
  },

  movePhotoFromClusterToPerson(clusterId, photoId, personId, name) {
    const db = readDB();
    if (!db.clusters) db.clusters = [];
    
    const clusterIndex = db.clusters.findIndex(c => c.id === clusterId);
    if (clusterIndex === -1) throw new Error('Cluster not found');
    const cluster = db.clusters[clusterIndex];
    
    const photoIndex = cluster.photos.findIndex(p => p.id === photoId);
    if (photoIndex === -1) throw new Error('Photo not found in cluster');
    const photoObj = cluster.photos[photoIndex];
    
    let person;
    if (personId) {
      person = db.persons.find(p => p.id === personId);
      if (!person) throw new Error('Person not found');
    } else {
      // Check if a person with that name already exists (case-insensitive) to merge them
      person = db.persons.find(p => p.name.toLowerCase() === name.trim().toLowerCase());
      if (!person) {
        person = {
          id: Date.now().toString(36) + Math.random().toString(36).substr(2, 5),
          name: name.trim(),
          gender: cluster.gender || 'Unknown',
          created_at: new Date().toISOString()
        };
        db.persons.push(person);
      }
    }
    
    // Copy the crop image from crops to uploads directory for enrollment
    const cropPath = path.join(CROPS_DIR, photoObj.filename);
    const uploadFilename = `enrolled_${Date.now()}_${photoObj.filename}`;
    const uploadPath = path.join(UPLOADS_DIR, uploadFilename);
    
    try {
      if (fs.existsSync(cropPath)) {
        fs.copyFileSync(cropPath, uploadPath);
      }
    } catch (err) {
      console.error('Failed to copy crop file to uploads during manual photo move:', err);
    }
    
    // Enroll the photo
    const newPhoto = {
      id: Date.now().toString(36) + Math.random().toString(36).substr(2, 4),
      person_id: person.id,
      filename: uploadFilename,
      embedding: photoObj.embedding
    };
    db.photos.push(newPhoto);
    
    // Remove the photo from the cluster
    cluster.photos.splice(photoIndex, 1);
    
    // If the cluster has no photos left, remove the cluster
    if (cluster.photos.length === 0) {
      db.clusters.splice(clusterIndex, 1);
    }
    
    writeDB(db);
    return newPhoto;
  },

  moveEventPhotoToPerson(eventId, personId, name, embedding) {
    const db = readDB();
    const event = db.events.find(e => e.id === eventId);
    if (!event) throw new Error('Event not found');
    
    let person;
    if (personId) {
      person = db.persons.find(p => p.id === personId);
      if (!person) throw new Error('Person not found');
    } else {
      person = db.persons.find(p => p.name.toLowerCase() === name.trim().toLowerCase());
      if (!person) {
        person = {
          id: Date.now().toString(36) + Math.random().toString(36).substr(2, 5),
          name: name.trim(),
          gender: 'Unknown',
          created_at: new Date().toISOString()
        };
        db.persons.push(person);
      }
    }
    
    // Check if the photo is in a cluster
    let photoEmbedding = embedding;
    const clusterId = event.person_id;
    if (clusterId && clusterId !== 'UNKNOWN') {
      const clusterIndex = db.clusters.findIndex(c => c.id === clusterId);
      if (clusterIndex !== -1) {
        const cluster = db.clusters[clusterIndex];
        const photoIndex = cluster.photos.findIndex(p => p.filename === event.crop_filename);
        if (photoIndex !== -1) {
          const photoObj = cluster.photos[photoIndex];
          if (!photoEmbedding) {
            photoEmbedding = photoObj.embedding;
          }
          // Remove the photo from the cluster
          cluster.photos.splice(photoIndex, 1);
          if (cluster.photos.length === 0) {
            db.clusters.splice(clusterIndex, 1);
          }
        }
      }
    }
    
    if (!photoEmbedding) {
      throw new Error('Face embedding not found. Please try again.');
    }
    
    // Copy the crop image from crops to uploads directory for enrollment
    const cropPath = path.join(CROPS_DIR, event.crop_filename);
    const uploadFilename = `enrolled_${Date.now()}_${event.crop_filename}`;
    const uploadPath = path.join(UPLOADS_DIR, uploadFilename);
    
    try {
      if (fs.existsSync(cropPath)) {
        fs.copyFileSync(cropPath, uploadPath);
      }
    } catch (err) {
      console.error('Failed to copy crop file to uploads during event photo move:', err);
    }
    
    // Enroll the photo
    const newPhoto = {
      id: Date.now().toString(36) + Math.random().toString(36).substr(2, 4),
      person_id: person.id,
      filename: uploadFilename,
      embedding: photoEmbedding
    };
    db.photos.push(newPhoto);
    
    // Update the event itself to make it a KNOWN event!
    event.person_id = person.id;
    event.person_name = person.name;
    event.is_known = true;
    event.score = 1.0;
    
    writeDB(db);
    return event;
  },

  deleteCluster(clusterId) {
    const db = readDB();
    if (!db.clusters) db.clusters = [];
    
    const cluster = db.clusters.find(c => c.id === clusterId);
    if (!cluster) return false;
    
    cluster.photos.forEach(ph => {
      const cropPath = path.join(CROPS_DIR, ph.filename);
      if (fs.existsSync(cropPath)) {
        try { fs.unlinkSync(cropPath); } catch (e) {}
      }
    });
    
    db.clusters = db.clusters.filter(c => c.id !== clusterId);
    writeDB(db);
    return true;
  },

  deduplicateAutoPhotos() {
    const db = readDB();
    let removedCount = 0;
    
    db.persons.forEach(person => {
      const allPhotos = db.photos.filter(p => p.person_id === person.id);
      const manualPhotos = allPhotos.filter(p => !p.filename.startsWith('auto_'));
      const autoPhotos = allPhotos.filter(p => p.filename.startsWith('auto_'));
      
      if (autoPhotos.length === 0) return;
      
      const keptAutoPhotos = [];
      const manualEmbs = manualPhotos.filter(p => p.embedding).map(p => p.embedding);
      
      for (const autoP of autoPhotos) {
        if (!autoP.embedding) {
          const filePath = path.join(UPLOADS_DIR, autoP.filename);
          if (fs.existsSync(filePath)) { try { fs.unlinkSync(filePath); } catch (e) {} }
          db.photos = db.photos.filter(p => p.id !== autoP.id);
          removedCount++;
          continue;
        }
        
        if (keptAutoPhotos.length >= 10) {
          const filePath = path.join(UPLOADS_DIR, autoP.filename);
          if (fs.existsSync(filePath)) { try { fs.unlinkSync(filePath); } catch (e) {} }
          db.photos = db.photos.filter(p => p.id !== autoP.id);
          removedCount++;
          continue;
        }
        
        const referenceEmbs = [...manualEmbs, ...keptAutoPhotos.map(p => p.embedding)];
        let isTooSimilar = false;
        for (const refEmb of referenceEmbs) {
          const sim = dotProduct(autoP.embedding, refEmb);
          if (sim >= 0.88) {
            isTooSimilar = true;
            break;
          }
        }
        
        if (isTooSimilar) {
          const filePath = path.join(UPLOADS_DIR, autoP.filename);
          if (fs.existsSync(filePath)) { try { fs.unlinkSync(filePath); } catch (e) {} }
          db.photos = db.photos.filter(p => p.id !== autoP.id);
          removedCount++;
        } else {
          keptAutoPhotos.push(autoP);
        }
      }
    });
    
    if (fs.existsSync(CROPS_DIR)) {
      const referencedCrops = new Set();
      (db.events || []).forEach(e => { if (e.crop_filename) referencedCrops.add(e.crop_filename); });
      (db.clusters || []).forEach(c => {
        (c.photos || []).forEach(p => { if (p.filename) referencedCrops.add(p.filename); });
      });
      
      try {
        const cropFiles = fs.readdirSync(CROPS_DIR);
        cropFiles.forEach(file => {
          if (!referencedCrops.has(file)) {
            const filePath = path.join(CROPS_DIR, file);
            try { fs.unlinkSync(filePath); } catch (e) {}
          }
        });
      } catch (err) {
        console.error('[DB Maintenance] Error cleaning up orphaned crops:', err);
      }
    }
    
    if (removedCount > 0) {
      writeDB(db);
      console.log(`[DB Maintenance] Cleaned up ${removedCount} duplicate auto-learned photos.`);
    }
    return removedCount;
  },

  resetAll() {
    const cleanData = {
      persons: [],
      photos: [],
      events: [],
      cameras: [],
      clusters: [],
      cluster_counter: 0
    };
    writeDB(cleanData);
    
    if (fs.existsSync(UPLOADS_DIR)) {
      try {
        const uploadFiles = fs.readdirSync(UPLOADS_DIR);
        uploadFiles.forEach(f => {
          try { fs.unlinkSync(path.join(UPLOADS_DIR, f)); } catch (e) {}
        });
      } catch (e) {}
    }
    
    if (fs.existsSync(CROPS_DIR)) {
      try {
        const cropFiles = fs.readdirSync(CROPS_DIR);
        cropFiles.forEach(f => {
          try { fs.unlinkSync(path.join(CROPS_DIR, f)); } catch (e) {}
        });
      } catch (e) {}
    }
    
    console.log('[DB Reset] Successfully wiped all persons, photos, events, cameras, and clusters.');
    return true;
  }
};
