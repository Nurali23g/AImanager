-- =========================
-- CLIENTS
-- =========================
CREATE TABLE IF NOT EXISTS clients ( 
    id SERIAL PRIMARY KEY,
    phone VARCHAR(20) UNIQUE NOT NULL,
    parent_name TEXT,
    preferred_language TEXT DEFAULT 'kz',
    wants_offline BOOLEAN,
    call_time_preference TEXT,
    ai_blocked BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

ALTER TABLE clients DROP COLUMN IF EXISTS region;

ALTER TABLE clients ADD COLUMN IF NOT EXISTS parent_name TEXT;
ALTER TABLE clients ADD COLUMN IF NOT EXISTS preferred_language TEXT DEFAULT 'kz';
ALTER TABLE clients ADD COLUMN IF NOT EXISTS wants_offline BOOLEAN;
ALTER TABLE clients ADD COLUMN IF NOT EXISTS call_time_preference TEXT;
ALTER TABLE clients ADD COLUMN IF NOT EXISTS ai_blocked BOOLEAN DEFAULT FALSE;
ALTER TABLE clients ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW();
ALTER TABLE clients ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW();


-- =========================
-- CONVERSATIONS
-- =========================
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    chat_id TEXT UNIQUE NOT NULL,
    current_student_id INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

ALTER TABLE conversations DROP COLUMN IF EXISTS last_question_key;
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS current_student_id INTEGER;
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW();
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW();


-- =========================
-- STUDENTS
-- =========================
CREATE TABLE IF NOT EXISTS students (
    id SERIAL PRIMARY KEY,
    client_id INTEGER NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    child_order INTEGER,
    relation_label TEXT,
    student_name TEXT,
    grade INTEGER,
    study_format TEXT,
    study_language TEXT,
    education_level TEXT,
    goal TEXT,
    course_interest TEXT,
    target_school TEXT,
    progress_notes TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

ALTER TABLE students ADD COLUMN IF NOT EXISTS child_order INTEGER;
ALTER TABLE students ADD COLUMN IF NOT EXISTS relation_label TEXT;
ALTER TABLE students ADD COLUMN IF NOT EXISTS student_name TEXT;
ALTER TABLE students ADD COLUMN IF NOT EXISTS grade INTEGER;
ALTER TABLE students ADD COLUMN IF NOT EXISTS study_format TEXT;
ALTER TABLE students ADD COLUMN IF NOT EXISTS study_language TEXT;
ALTER TABLE students ADD COLUMN IF NOT EXISTS education_level TEXT;
ALTER TABLE students ADD COLUMN IF NOT EXISTS goal TEXT;
ALTER TABLE students ADD COLUMN IF NOT EXISTS course_interest TEXT;
ALTER TABLE students ADD COLUMN IF NOT EXISTS target_school TEXT;
ALTER TABLE students ADD COLUMN IF NOT EXISTS progress_notes TEXT;
ALTER TABLE students ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW();
ALTER TABLE students ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW();


-- =========================
-- MESSAGES
-- =========================
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    message_text TEXT NOT NULL,
    token_estimate INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);


-- =========================
-- BOT SETTINGS
-- =========================
CREATE TABLE IF NOT EXISTS bot_settings (
    id INTEGER PRIMARY KEY DEFAULT 1,
    global_bot_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO bot_settings (id, global_bot_enabled)
VALUES (1, TRUE)
ON CONFLICT (id) DO NOTHING;


-- =========================
-- OPTIONAL INDEXES
-- =========================
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_students_client_id ON students(client_id);
CREATE INDEX IF NOT EXISTS idx_clients_phone ON clients(phone);
