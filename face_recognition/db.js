const fs = require('fs');
const path = require('path');
const { DatabaseSync } = require('node:sqlite');

const DATA_DIR = path.join(__dirname, 'data');
const SQLITE_FILE = path.join(DATA_DIR, 'face_recognition.db');
const UPLOADS_DIR = path.join(DATA_DIR, 'uploads');
const CROPS_DIR = path.join(DATA_DIR, 'crops');

fs.mkdirSync(DATA_DIR, { recursive: true });
fs.mkdirSync(UPLOADS_DIR, { recursive: true });
fs.mkdirSync(CROPS_DIR, { recursive: true });

const sqlite = new DatabaseSync(SQLITE_FILE, {
  enableForeignKeyConstraints: true,
  timeout: 5000,
  defensive: true
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
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS photos (
  id TEXT PRIMARY KEY,
  person_id TEXT NOT NULL,
  filename TEXT,
  embedding BLOB,
  created_at TEXT NOT NULL,
  FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY,
  timestamp TEXT NOT NULL,
  person_id TEXT,
  person_name TEXT,
  score REAL NOT NULL DEFAULT 0,
  crop_filename TEXT,
  is_known INTEGER NOT NULL DEFAULT 0,
  camera_id TEXT,
  camera_name TEXT
);

CREATE TABLE IF NOT EXISTS cameras (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  rtsp_url TEXT NOT NULL,
  is_active INTEGER NOT NULL DEFAULT 0,
  line_crossing_enabled INTEGER NOT NULL DEFAULT 0,
  line_y REAL NOT NULL DEFAULT 0.6,
  line_direction TEXT NOT NULL DEFAULT 'in',
  line_x_start REAL NOT NULL DEFAULT 0,
  line_x_end REAL NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS clusters (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  gender TEXT NOT NULL DEFAULT 'Unknown',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cluster_photos (
  id TEXT PRIMARY KEY,
  cluster_id TEXT NOT NULL,
  filename TEXT,
  embedding BLOB,
  gender TEXT NOT NULL DEFAULT 'Unknown',
  FOREIGN KEY (cluster_id) REFERENCES clusters(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS app_settings (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  threshold REAL NOT NULL DEFAULT 0.50,
  dis_type INTEGER NOT NULL DEFAULT 0,
  cluster_counter INTEGER NOT NULL DEFAULT 0
);

INSERT OR IGNORE INTO app_settings(id, threshold, dis_type, cluster_counter)
VALUES (1, 0.50, 0, 0);

CREATE INDEX IF NOT EXISTS idx_photos_person ON photos(person_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_events_person ON events(person_id);
CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera_id);
CREATE INDEX IF NOT EXISTS idx_cluster_photos_cluster ON cluster_photos(cluster_id);
`);

function newId(prefix = '') {
  return prefix + Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}

function encodeEmbedding(embedding) {
  if (!Array.isArray(embedding) && !(embedding instanceof Float32Array)) return null;
  const values = Float32Array.from(embedding);
  return Buffer.from(values.buffer, values.byteOffset, values.byteLength);
}

function decodeEmbedding(value) {
  if (value === null || value === undefined) return null;
  const buffer = Buffer.isBuffer(value) ? value : Buffer.from(value);
  if (buffer.length === 0 || buffer.length % 4 !== 0) return null;
  const copy = new Uint8Array(buffer.length);
  copy.set(buffer);
  return Array.from(new Float32Array(copy.buffer));
}

function personRow(row) {
  if (!row) return null;
  return {
    id: row.id,
    name: row.name,
    gender: row.gender,
    created_at: row.created_at
  };
}

function photoRow(row) {
  if (!row) return null;
  return {
    id: row.id,
    person_id: row.person_id,
    filename: row.filename,
    embedding: decodeEmbedding(row.embedding)
  };
}

function eventRow(row) {
  if (!row) return null;
  return {
    id: row.id,
    timestamp: row.timestamp,
    person_id: row.person_id,
    person_name: row.person_name,
    score: Number(row.score || 0),
    crop_filename: row.crop_filename,
    is_known: !!row.is_known,
    camera_id: row.camera_id,
    camera_name: row.camera_name
  };
}

function cameraRow(row) {
  if (!row) return null;
  return {
    id: row.id,
    name: row.name,
    rtsp_url: row.rtsp_url,
    is_active: !!row.is_active,
    line_crossing_enabled: !!row.line_crossing_enabled,
    line_y: Number(row.line_y),
    line_direction: row.line_direction,
    line_x_start: Number(row.line_x_start),
    line_x_end: Number(row.line_x_end),
    created_at: row.created_at
  };
}

function clusterPhotoRow(row) {
  return {
    ...(row.id ? { id: row.id } : {}),
    filename: row.filename,
    embedding: decodeEmbedding(row.embedding),
    gender: row.gender || 'Unknown'
  };
}

function getCluster(id) {
  const row = sqlite.prepare(
    'SELECT id,name,gender,created_at FROM clusters WHERE id=?'
  ).get(id);
  if (!row) return null;

  const photos = sqlite.prepare(
    'SELECT id,filename,embedding,gender FROM cluster_photos WHERE cluster_id=? ORDER BY rowid'
  ).all(id).map(clusterPhotoRow);

  return { ...row, photos };
}

function getSettings() {
  const row = sqlite.prepare(
    'SELECT threshold,dis_type,cluster_counter FROM app_settings WHERE id=1'
  ).get();
  return {
    threshold: Number(row?.threshold ?? 0.50),
    dis_type: Number(row?.dis_type ?? 0),
    cluster_counter: Number(row?.cluster_counter ?? 0)
  };
}

function setSetting(key, value) {
  if (key === 'threshold') {
    sqlite.prepare('UPDATE app_settings SET threshold=? WHERE id=1').run(Number(value));
  } else if (key === 'dis_type') {
    sqlite.prepare('UPDATE app_settings SET dis_type=? WHERE id=1').run(Number(value));
  } else if (key === 'cluster_counter') {
    sqlite.prepare('UPDATE app_settings SET cluster_counter=? WHERE id=1').run(Number(value));
  } else {
    throw new Error('Unknown database setting: ' + key);
  }
}

function updateClusterCounter() {
  const row = sqlite.prepare(
    'SELECT cluster_counter FROM app_settings WHERE id=1'
  ).get();
  const next = Number(row?.cluster_counter || 0) + 1;
  sqlite.prepare('UPDATE app_settings SET cluster_counter=? WHERE id=1').run(next);
  return next;
}

function deleteFile(dir, filename) {
  if (!filename) return;
  const filePath = path.join(dir, filename);
  if (fs.existsSync(filePath)) {
    try { fs.unlinkSync(filePath); } catch (_) {}
  }
}

// Math helpers for vector comparisons with L2 normalization.
function dotProduct(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
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
  if (!a || !b || a.length !== b.length) return 2.0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);
  if (normA === 0 || normB === 0) return 2.0;

  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = (a[i] / normA) - (b[i] / normB);
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

const api = {
  DATA_DIR,
  SQLITE_FILE,
  UPLOADS_DIR,
  CROPS_DIR,

  getSettings,

  setSetting,

  getPersons() {
    const persons = sqlite.prepare(
      'SELECT id,name,gender,created_at FROM persons ORDER BY rowid'
    ).all();

    const photoStmt = sqlite.prepare(
      'SELECT id,filename FROM photos WHERE person_id=? ORDER BY rowid'
    );

    return persons.map(person => ({
      ...personRow(person),
      photos: photoStmt.all(person.id).map(p => ({
        id: p.id,
        filename: p.filename
      }))
    }));
  },

  getPerson(id) {
    const person = sqlite.prepare(
      'SELECT id,name,gender,created_at FROM persons WHERE id=?'
    ).get(id);
    if (!person) return null;

    const photos = sqlite.prepare(
      'SELECT id,filename FROM photos WHERE person_id=? ORDER BY rowid'
    ).all(id);

    return {
      ...personRow(person),
      photos: photos.map(p => ({ id: p.id, filename: p.filename }))
    };
  },

  addPerson(name, gender = 'Unknown') {
    const person = {
      id: newId(),
      name,
      gender: gender || 'Unknown',
      created_at: new Date().toISOString()
    };

    sqlite.prepare(
      'INSERT INTO persons(id,name,gender,created_at) VALUES(?,?,?,?)'
    ).run(person.id, person.name, person.gender, person.created_at);

    return { ...person, photos: [] };
  },

  deletePerson(id) {
    const photos = sqlite.prepare(
      'SELECT filename FROM photos WHERE person_id=?'
    ).all(id);

    sqlite.prepare('DELETE FROM persons WHERE id=?').run(id);

    for (const photo of photos) deleteFile(UPLOADS_DIR, photo.filename);
    return true;
  },

  addPhoto(personId, filename, embedding) {
    const exists = sqlite.prepare(
      'SELECT id FROM persons WHERE id=?'
    ).get(personId);
    if (!exists) throw new Error('Person not found');

    const photo = {
      id: newId(),
      person_id: personId,
      filename,
      embedding
    };

    sqlite.prepare(
      'INSERT INTO photos(id,person_id,filename,embedding,created_at) VALUES(?,?,?,?,?)'
    ).run(
      photo.id,
      photo.person_id,
      photo.filename || null,
      encodeEmbedding(photo.embedding),
      new Date().toISOString()
    );

    return photo;
  },

  deletePhoto(photoId) {
    const photo = sqlite.prepare(
      'SELECT filename FROM photos WHERE id=?'
    ).get(photoId);
    if (!photo) return false;

    sqlite.prepare('DELETE FROM photos WHERE id=?').run(photoId);
    deleteFile(UPLOADS_DIR, photo.filename);
    return true;
  },

  getAllCandidates() {
    const rows = sqlite.prepare(`
      SELECT p.id AS person_id, p.name, ph.embedding
      FROM persons p
      JOIN photos ph ON ph.person_id=p.id
      ORDER BY p.rowid, ph.rowid
    `).all();

    const candidates = new Map();
    for (const row of rows) {
      const embedding = decodeEmbedding(row.embedding);
      if (!embedding) continue;

      if (!candidates.has(row.person_id)) {
        candidates.set(row.person_id, {
          person_id: row.person_id,
          name: row.name,
          embeddings: []
        });
      }
      candidates.get(row.person_id).embeddings.push(embedding);
    }
    return Array.from(candidates.values()).filter(c => c.embeddings.length);
  },

  getCameras() {
    return sqlite.prepare(
      'SELECT id,name,rtsp_url,is_active,line_crossing_enabled,line_y,line_direction,line_x_start,line_x_end,created_at FROM cameras ORDER BY rowid'
    ).all().map(cameraRow);
  },

  getCamera(id) {
    return cameraRow(sqlite.prepare(
      'SELECT id,name,rtsp_url,is_active,line_crossing_enabled,line_y,line_direction,line_x_start,line_x_end,created_at FROM cameras WHERE id=?'
    ).get(id));
  },

  updateCamera(id, updates = {}) {
    const camera = api.getCamera(id);
    if (!camera) return null;

    const merged = { ...camera, ...updates };
    sqlite.prepare(`
      UPDATE cameras
      SET name=?,rtsp_url=?,is_active=?,line_crossing_enabled=?,
          line_y=?,line_direction=?,line_x_start=?,line_x_end=?
      WHERE id=?
    `).run(
      merged.name,
      merged.rtsp_url,
      merged.is_active ? 1 : 0,
      merged.line_crossing_enabled ? 1 : 0,
      Number(merged.line_y ?? 0.6),
      merged.line_direction || 'in',
      Number(merged.line_x_start ?? 0),
      Number(merged.line_x_end ?? 1),
      id
    );

    return api.getCamera(id);
  },

  addCamera(name, rtspUrl) {
    const camera = {
      id: 'cam_' + newId(),
      name,
      rtsp_url: rtspUrl,
      is_active: false,
      line_crossing_enabled: false,
      line_y: 0.6,
      line_direction: 'in',
      line_x_start: 0,
      line_x_end: 1,
      created_at: new Date().toISOString()
    };

    sqlite.prepare(`
      INSERT INTO cameras(
        id,name,rtsp_url,is_active,line_crossing_enabled,
        line_y,line_direction,line_x_start,line_x_end,created_at
      ) VALUES(?,?,?,?,?,?,?,?,?,?)
    `).run(
      camera.id, camera.name, camera.rtsp_url, 0, 0,
      camera.line_y, camera.line_direction,
      camera.line_x_start, camera.line_x_end, camera.created_at
    );

    return camera;
  },

  deleteCamera(id) {
    sqlite.prepare('DELETE FROM cameras WHERE id=?').run(id);
    return true;
  },

  updateCameraStatus(id, isActive) {
    sqlite.prepare(
      'UPDATE cameras SET is_active=? WHERE id=?'
    ).run(isActive ? 1 : 0, id);
    return api.getCamera(id);
  },

  updateCameraLineSettings(id, enabled, lineY, direction, lineXStart, lineXEnd) {
    const camera = api.getCamera(id);
    if (!camera) return null;

    const next = {
      enabled: enabled !== undefined ? !!enabled : camera.line_crossing_enabled,
      lineY: lineY !== undefined ? parseFloat(lineY) : camera.line_y,
      direction: direction !== undefined ? direction : camera.line_direction,
      lineXStart: lineXStart !== undefined ? parseFloat(lineXStart) : camera.line_x_start,
      lineXEnd: lineXEnd !== undefined ? parseFloat(lineXEnd) : camera.line_x_end
    };

    sqlite.prepare(`
      UPDATE cameras
      SET line_crossing_enabled=?,line_y=?,line_direction=?,line_x_start=?,line_x_end=?
      WHERE id=?
    `).run(
      next.enabled ? 1 : 0,
      next.lineY,
      next.direction,
      next.lineXStart,
      next.lineXEnd,
      id
    );

    return api.getCamera(id);
  },

  getEvents(limit = 100) {
    const safeLimit = Math.max(1, Math.min(1000, Number(limit) || 100));
    return sqlite.prepare(`
      SELECT id,timestamp,person_id,person_name,score,crop_filename,
             is_known,camera_id,camera_name
      FROM events
      ORDER BY timestamp DESC
      LIMIT ?
    `).all(safeLimit).map(eventRow);
  },

  addEvent(personId, personName, score, cropFilename, isKnown, cameraId = null, cameraName = 'Manual Upload') {
    const event = {
      id: newId(),
      timestamp: new Date().toISOString(),
      person_id: personId,
      person_name: personName,
      score: Number(score || 0),
      crop_filename: cropFilename,
      is_known: !!isKnown,
      camera_id: cameraId,
      camera_name: cameraName
    };

    sqlite.prepare(`
      INSERT INTO events(
        id,timestamp,person_id,person_name,score,crop_filename,
        is_known,camera_id,camera_name
      ) VALUES(?,?,?,?,?,?,?,?,?)
    `).run(
      event.id, event.timestamp, event.person_id, event.person_name,
      event.score, event.crop_filename, event.is_known ? 1 : 0,
      event.camera_id, event.camera_name
    );

    const oldEvents = sqlite.prepare(`
      SELECT id,crop_filename FROM events
      ORDER BY timestamp DESC
      LIMIT -1 OFFSET 1000
    `).all();

    if (oldEvents.length) {
      const deleteStmt = sqlite.prepare('DELETE FROM events WHERE id=?');
      for (const old of oldEvents) {
        deleteStmt.run(old.id);
        deleteFile(CROPS_DIR, old.crop_filename);
      }
    }

    return event;
  },

  updateEvent(eventId, updates = {}) {
    const event = sqlite.prepare(
      'SELECT id,timestamp,person_id,person_name,score,crop_filename,is_known,camera_id,camera_name FROM events WHERE id=?'
    ).get(eventId);
    if (!event) return null;

    const merged = { ...eventRow(event), ...updates };
    sqlite.prepare(`
      UPDATE events
      SET timestamp=?,person_id=?,person_name=?,score=?,crop_filename=?,
          is_known=?,camera_id=?,camera_name=?
      WHERE id=?
    `).run(
      merged.timestamp,
      merged.person_id,
      merged.person_name,
      Number(merged.score || 0),
      merged.crop_filename,
      merged.is_known ? 1 : 0,
      merged.camera_id,
      merged.camera_name,
      eventId
    );

    return eventRow(sqlite.prepare(
      'SELECT id,timestamp,person_id,person_name,score,crop_filename,is_known,camera_id,camera_name FROM events WHERE id=?'
    ).get(eventId));
  },

  clearEvents() {
    const events = sqlite.prepare(
      'SELECT crop_filename FROM events WHERE crop_filename IS NOT NULL'
    ).all();

    sqlite.exec('DELETE FROM events');
    for (const event of events) deleteFile(CROPS_DIR, event.crop_filename);
    return true;
  },

  getClusters() {
    const rows = sqlite.prepare(
      'SELECT id,name,gender,created_at FROM clusters ORDER BY rowid'
    ).all();

    const photoStmt = sqlite.prepare(
      'SELECT id,filename,embedding,gender FROM cluster_photos WHERE cluster_id=? ORDER BY rowid'
    );

    return rows.map(row => ({
      ...row,
      photos: photoStmt.all(row.id).map(clusterPhotoRow)
    }));
  },

  addFaceToCluster(embedding, cropFilename, gender, settings) {
    const disType = Number(settings.dis_type);
    const threshold = disType === 0
      ? Math.max(0.58, Number(settings.threshold) + 0.02)
      : Math.min(0.95, Number(settings.threshold) - 0.15);

    const clusters = api.getClusters();
    let bestCluster = null;
    let bestScore = disType === 0 ? -1.0 : Infinity;

    for (const cluster of clusters) {
      if (!cluster.photos.length) continue;

      const scores = cluster.photos.map(photo =>
        disType === 0
          ? dotProduct(embedding, photo.embedding)
          : l2Distance(embedding, photo.embedding)
      );
      const avgScore = scores.reduce((sum, value) => sum + value, 0) / scores.length;

      if (disType === 0) {
        if (avgScore >= threshold && avgScore > bestScore) {
          bestScore = avgScore;
          bestCluster = cluster;
        }
      } else if (avgScore <= threshold && avgScore < bestScore) {
        bestScore = avgScore;
        bestCluster = cluster;
      }
    }

    if (bestCluster) {
      sqlite.prepare(`
        INSERT INTO cluster_photos(id,cluster_id,filename,embedding,gender)
        VALUES(?,?,?,?,?)
      `).run(
        newId('cphoto_'),
        bestCluster.id,
        cropFilename,
        encodeEmbedding(embedding),
        gender || 'Unknown'
      );

      const genderRows = sqlite.prepare(
        'SELECT gender FROM cluster_photos WHERE cluster_id=?'
      ).all(bestCluster.id);

      let male = 0;
      let female = 0;
      for (const row of genderRows) {
        if (row.gender === 'Male') male++;
        if (row.gender === 'Female') female++;
      }

      const finalGender = male > female ? 'Male' : female > male ? 'Female' : 'Unknown';
      const baseName = bestCluster.name.split(' (')[0];
      const finalName = baseName + (finalGender !== 'Unknown' ? ' (' + finalGender + ')' : '');

      sqlite.prepare(
        'UPDATE clusters SET gender=?,name=? WHERE id=?'
      ).run(finalGender, finalName, bestCluster.id);

      return {
        id: bestCluster.id,
        name: finalName,
        gender: finalGender,
        isNew: false
      };
    }

    const nextNum = updateClusterCounter();
    const finalGender = gender || 'Unknown';
    const suffix = finalGender !== 'Unknown' ? ' (' + finalGender + ')' : '';
    const clusterId = 'cluster_' + newId();

    sqlite.prepare(
      'INSERT INTO clusters(id,name,gender,created_at) VALUES(?,?,?,?)'
    ).run(
      clusterId,
      'Profile #' + nextNum + suffix,
      finalGender,
      new Date().toISOString()
    );

    sqlite.prepare(`
      INSERT INTO cluster_photos(id,cluster_id,filename,embedding,gender)
      VALUES(?,?,?,?,?)
    `).run(
      newId('cphoto_'),
      clusterId,
      cropFilename,
      encodeEmbedding(embedding),
      finalGender
    );

    return {
      id: clusterId,
      name: 'Profile #' + nextNum + suffix,
      gender: finalGender,
      isNew: true
    };
  },

  enrollCluster(clusterId, name) {
    const cluster = getCluster(clusterId);
    if (!cluster) throw new Error('Cluster not found');

    let person = sqlite.prepare(
      'SELECT id,name,gender,created_at FROM persons WHERE lower(name)=lower(?) LIMIT 1'
    ).get(name.trim());

    const isNew = !person;
    if (!person) {
      person = {
        id: newId(),
        name: name.trim(),
        gender: cluster.gender || 'Unknown',
        created_at: new Date().toISOString()
      };
      sqlite.prepare(
        'INSERT INTO persons(id,name,gender,created_at) VALUES(?,?,?,?)'
      ).run(person.id, person.name, person.gender, person.created_at);
    } else if (
      cluster.gender &&
      cluster.gender !== 'Unknown' &&
      (!person.gender || person.gender === 'Unknown')
    ) {
      sqlite.prepare('UPDATE persons SET gender=? WHERE id=?')
        .run(cluster.gender, person.id);
      person.gender = cluster.gender;
    }

    const insertPhoto = sqlite.prepare(`
      INSERT INTO photos(id,person_id,filename,embedding,created_at)
      VALUES(?,?,?,?,?)
    `);

    for (const photo of cluster.photos) {
      const uploadFilename = 'enrolled_' + Date.now() + '_' + path.basename(photo.filename || 'face.jpg');
      const cropPath = path.join(CROPS_DIR, photo.filename || '');
      const uploadPath = path.join(UPLOADS_DIR, uploadFilename);

      try {
        if (photo.filename && fs.existsSync(cropPath)) {
          fs.copyFileSync(cropPath, uploadPath);
        }
      } catch (err) {
        console.error('Failed to copy crop file to uploads during enrollment:', err);
      }

      insertPhoto.run(
        newId(),
        person.id,
        uploadFilename,
        encodeEmbedding(photo.embedding),
        new Date().toISOString()
      );
    }

    sqlite.prepare(`
      UPDATE events
      SET person_id=?,person_name=?,is_known=1
      WHERE person_id=?
    `).run(
      person.id,
      isNew ? person.name : person.name + (
        person.gender && person.gender !== 'Unknown' ? ' (' + person.gender + ')' : ''
      ),
      clusterId
    );

    sqlite.prepare('DELETE FROM clusters WHERE id=?').run(clusterId);
    return api.getPerson(person.id);
  },

  movePhotoFromClusterToPerson(clusterId, photoId, personId, name) {
    const cluster = getCluster(clusterId);
    if (!cluster) throw new Error('Cluster not found');

    const photo = cluster.photos.find(p => p.id === photoId);
    if (!photo) throw new Error('Photo not found in cluster');

    let person;
    if (personId) {
      person = sqlite.prepare(
        'SELECT id,name,gender,created_at FROM persons WHERE id=?'
      ).get(personId);
      if (!person) throw new Error('Person not found');
    } else {
      person = sqlite.prepare(
        'SELECT id,name,gender,created_at FROM persons WHERE lower(name)=lower(?) LIMIT 1'
      ).get(name.trim());

      if (!person) {
        person = {
          id: newId(),
          name: name.trim(),
          gender: cluster.gender || 'Unknown',
          created_at: new Date().toISOString()
        };
        sqlite.prepare(
          'INSERT INTO persons(id,name,gender,created_at) VALUES(?,?,?,?)'
        ).run(person.id, person.name, person.gender, person.created_at);
      }
    }

    const uploadFilename = 'enrolled_' + Date.now() + '_' + path.basename(photo.filename || 'face.jpg');
    const cropPath = path.join(CROPS_DIR, photo.filename || '');
    const uploadPath = path.join(UPLOADS_DIR, uploadFilename);

    try {
      if (photo.filename && fs.existsSync(cropPath)) {
        fs.copyFileSync(cropPath, uploadPath);
      }
    } catch (err) {
      console.error('Failed to copy crop file to uploads during manual photo move:', err);
    }

    const newPhoto = {
      id: newId(),
      person_id: person.id,
      filename: uploadFilename,
      embedding: photo.embedding
    };

    sqlite.prepare(`
      INSERT INTO photos(id,person_id,filename,embedding,created_at)
      VALUES(?,?,?,?,?)
    `).run(
      newPhoto.id,
      newPhoto.person_id,
      newPhoto.filename,
      encodeEmbedding(newPhoto.embedding),
      new Date().toISOString()
    );

    sqlite.prepare('DELETE FROM cluster_photos WHERE id=?').run(photo.id);

    const remaining = sqlite.prepare(
      'SELECT COUNT(*) AS count FROM cluster_photos WHERE cluster_id=?'
    ).get(clusterId).count;

    if (Number(remaining) === 0) {
      sqlite.prepare('DELETE FROM clusters WHERE id=?').run(clusterId);
    }

    return newPhoto;
  },

  moveEventPhotoToPerson(eventId, personId, name, embedding) {
    const event = sqlite.prepare(
      'SELECT id,timestamp,person_id,person_name,score,crop_filename,is_known,camera_id,camera_name FROM events WHERE id=?'
    ).get(eventId);
    if (!event) throw new Error('Event not found');

    let person;
    if (personId) {
      person = sqlite.prepare(
        'SELECT id,name,gender,created_at FROM persons WHERE id=?'
      ).get(personId);
      if (!person) throw new Error('Person not found');
    } else {
      person = sqlite.prepare(
        'SELECT id,name,gender,created_at FROM persons WHERE lower(name)=lower(?) LIMIT 1'
      ).get(name.trim());

      if (!person) {
        person = {
          id: newId(),
          name: name.trim(),
          gender: 'Unknown',
          created_at: new Date().toISOString()
        };
        sqlite.prepare(
          'INSERT INTO persons(id,name,gender,created_at) VALUES(?,?,?,?)'
        ).run(person.id, person.name, person.gender, person.created_at);
      }
    }

    let photoEmbedding = embedding;
    const clusterId = event.person_id;

    if (clusterId && clusterId !== 'UNKNOWN') {
      const cluster = getCluster(clusterId);
      if (cluster) {
        const clusterPhoto = cluster.photos.find(p => p.filename === event.crop_filename);
        if (clusterPhoto) {
          if (!photoEmbedding) photoEmbedding = clusterPhoto.embedding;
          sqlite.prepare('DELETE FROM cluster_photos WHERE id=?').run(clusterPhoto.id);

          const remaining = sqlite.prepare(
            'SELECT COUNT(*) AS count FROM cluster_photos WHERE cluster_id=?'
          ).get(clusterId).count;
          if (Number(remaining) === 0) {
            sqlite.prepare('DELETE FROM clusters WHERE id=?').run(clusterId);
          }
        }
      }
    }

    if (!photoEmbedding) {
      throw new Error('Face embedding not found. Please try again.');
    }

    const uploadFilename = 'enrolled_' + Date.now() + '_' + path.basename(event.crop_filename || 'face.jpg');
    const cropPath = path.join(CROPS_DIR, event.crop_filename || '');
    const uploadPath = path.join(UPLOADS_DIR, uploadFilename);

    try {
      if (event.crop_filename && fs.existsSync(cropPath)) {
        fs.copyFileSync(cropPath, uploadPath);
      }
    } catch (err) {
      console.error('Failed to copy crop file to uploads during event photo move:', err);
    }

    sqlite.prepare(`
      INSERT INTO photos(id,person_id,filename,embedding,created_at)
      VALUES(?,?,?,?,?)
    `).run(
      newId(),
      person.id,
      uploadFilename,
      encodeEmbedding(photoEmbedding),
      new Date().toISOString()
    );

    sqlite.prepare(`
      UPDATE events
      SET person_id=?,person_name=?,is_known=1,score=1.0
      WHERE id=?
    `).run(person.id, person.name, eventId);

    return eventRow(sqlite.prepare(
      'SELECT id,timestamp,person_id,person_name,score,crop_filename,is_known,camera_id,camera_name FROM events WHERE id=?'
    ).get(eventId));
  },

  deleteCluster(clusterId) {
    const cluster = getCluster(clusterId);
    if (!cluster) return false;

    for (const photo of cluster.photos) deleteFile(CROPS_DIR, photo.filename);
    sqlite.prepare('DELETE FROM clusters WHERE id=?').run(clusterId);
    return true;
  },

  deduplicateAutoPhotos() {
    const persons = sqlite.prepare(
      'SELECT id FROM persons ORDER BY rowid'
    ).all();

    let removedCount = 0;

    for (const person of persons) {
      const photos = sqlite.prepare(`
        SELECT id,filename,embedding
        FROM photos
        WHERE person_id=?
        ORDER BY rowid
      `).all(person.id);

      const manual = photos.filter(p => p.filename && !p.filename.startsWith('auto_'));
      const auto = photos.filter(p => p.filename && p.filename.startsWith('auto_'));
      const kept = [];
      const manualEmbeddings = manual.map(p => decodeEmbedding(p.embedding)).filter(Boolean);

      for (const autoPhoto of auto) {
        const embedding = decodeEmbedding(autoPhoto.embedding);

        if (!embedding || kept.length >= 10) {
          deleteFile(UPLOADS_DIR, autoPhoto.filename);
          sqlite.prepare('DELETE FROM photos WHERE id=?').run(autoPhoto.id);
          removedCount++;
          continue;
        }

        let tooSimilar = false;
        const references = manualEmbeddings.concat(
          kept.map(p => p.embedding)
        );

        for (const ref of references) {
          if (dotProduct(embedding, ref) >= 0.88) {
            tooSimilar = true;
            break;
          }
        }

        if (tooSimilar) {
          deleteFile(UPLOADS_DIR, autoPhoto.filename);
          sqlite.prepare('DELETE FROM photos WHERE id=?').run(autoPhoto.id);
          removedCount++;
        } else {
          kept.push({ id: autoPhoto.id, embedding });
        }
      }
    }

    // Remove orphan crop files. Database is the source of truth for crop references.
    const referenced = new Set();
    sqlite.prepare(
      'SELECT crop_filename FROM events WHERE crop_filename IS NOT NULL'
    ).all().forEach(row => referenced.add(row.crop_filename));
    sqlite.prepare(
      'SELECT filename FROM cluster_photos WHERE filename IS NOT NULL'
    ).all().forEach(row => referenced.add(row.filename));

    if (fs.existsSync(CROPS_DIR)) {
      for (const file of fs.readdirSync(CROPS_DIR)) {
        if (!referenced.has(file)) deleteFile(CROPS_DIR, file);
      }
    }

    if (removedCount) {
      console.log('[DB Maintenance] Cleaned up ' + removedCount + ' duplicate auto-learned photos.');
    }

    return removedCount;
  },

  resetAll() {
    sqlite.exec(`
      DELETE FROM cluster_photos;
      DELETE FROM clusters;
      DELETE FROM photos;
      DELETE FROM persons;
      DELETE FROM events;
      DELETE FROM cameras;
      UPDATE app_settings SET threshold=0.50,dis_type=0,cluster_counter=0 WHERE id=1;
    `);

    for (const dir of [UPLOADS_DIR, CROPS_DIR]) {
      if (!fs.existsSync(dir)) continue;
      for (const file of fs.readdirSync(dir)) {
        deleteFile(dir, file);
      }
    }

    console.log('[DB Reset] Successfully wiped all persons, photos, events, cameras, and clusters.');
    return true;
  }
};

module.exports = api;
