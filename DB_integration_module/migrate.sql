CREATE TABLE IF NOT EXISTS call_records (
  id SERIAL PRIMARY KEY,
  conversation_file VARCHAR(255),
  phone_number VARCHAR(20),
  intent VARCHAR(20),
  plot_location TEXT,
  visit_date DATE,
  conversation_summary JSONB,
  faiss_ref TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  audio_data BYTEA
);

CREATE TABLE IF NOT EXISTS properties (
  property_id SERIAL PRIMARY KEY,
  location TEXT,
  price NUMERIC,
  area NUMERIC,
  property_type VARCHAR(50),
  raw_data JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_call_phone ON call_records (phone_number);
CREATE INDEX IF NOT EXISTS idx_call_intent ON call_records (intent);
CREATE INDEX IF NOT EXISTS idx_call_created_at ON call_records (created_at);