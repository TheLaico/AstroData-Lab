-- AstroData Lab - Extension pgvector, tablas vectoriales e indices.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS Embedding_Texto (
    id_embedding SERIAL PRIMARY KEY,
    id_doc INT NOT NULL REFERENCES Documento(id_doc) ON DELETE CASCADE,
    chunk_id INT NOT NULL CHECK (chunk_id >= 0),
    estrategia_chunking VARCHAR(40) NOT NULL CHECK (
        estrategia_chunking IN ('fixed', 'sentence', 'semantic', 'descripcion_objeto')
    ),
    vector vector(384) NOT NULL,
    modelo VARCHAR(160) NOT NULL,
    contenido_chunk TEXT,
    UNIQUE (id_doc, chunk_id, estrategia_chunking, modelo)
);

CREATE TABLE IF NOT EXISTS Embedding_Imagen (
    id_embedding SERIAL PRIMARY KEY,
    id_imagen INT NOT NULL REFERENCES Imagen(id_imagen) ON DELETE CASCADE,
    vector vector(512) NOT NULL,
    modelo VARCHAR(160) NOT NULL
);

CREATE TABLE IF NOT EXISTS Embedding_Consulta (
    id_embedding SERIAL PRIMARY KEY,
    id_consulta INT NOT NULL UNIQUE REFERENCES Consulta(id_consulta) ON DELETE CASCADE,
    vector vector(384) NOT NULL,
    modelo VARCHAR(160) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_embedding_texto_vector
ON Embedding_Texto USING ivfflat (vector vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_embedding_imagen_vector
ON Embedding_Imagen USING ivfflat (vector vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_embedding_consulta_vector
ON Embedding_Consulta USING ivfflat (vector vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_documento_objeto ON Documento(id_objeto);
CREATE INDEX IF NOT EXISTS idx_consulta_usuario ON Consulta(id_usuario);
CREATE INDEX IF NOT EXISTS idx_observacion_objeto ON Observacion(id_objeto);
