-- AstroData Lab - Esquema relacional base.

CREATE TABLE IF NOT EXISTS Usuario (
    id_usuario SERIAL PRIMARY KEY,
    nombre VARCHAR(120) NOT NULL,
    correo VARCHAR(160) NOT NULL UNIQUE,
    fecha_registro DATE NOT NULL DEFAULT CURRENT_DATE
);

CREATE TABLE IF NOT EXISTS Consulta (
    id_consulta SERIAL PRIMARY KEY,
    texto_pregunta TEXT NOT NULL,
    fecha TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    id_usuario INT NOT NULL REFERENCES Usuario(id_usuario) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS Objeto_Astronomico (
    id_objeto SERIAL PRIMARY KEY,
    nombre VARCHAR(160) NOT NULL,
    descripcion_cientifica TEXT
);

CREATE TABLE IF NOT EXISTS Documento (
    id_doc SERIAL PRIMARY KEY,
    titulo VARCHAR(220) NOT NULL,
    idioma VARCHAR(16),
    fecha DATE,
    fuente TEXT,
    contenido_texto TEXT,
    id_objeto INT REFERENCES Objeto_Astronomico(id_objeto) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS Imagen (
    id_imagen SERIAL PRIMARY KEY,
    ruta_archivo TEXT NOT NULL,
    descripcion TEXT,
    etiquetas TEXT,
    id_doc INT REFERENCES Documento(id_doc) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS Resultado (
    id_resultado SERIAL PRIMARY KEY,
    descripcion_resultado TEXT,
    relevancia DOUBLE PRECISION NOT NULL CHECK (relevancia >= 0 AND relevancia <= 1),
    id_consulta INT NOT NULL REFERENCES Consulta(id_consulta) ON DELETE CASCADE,
    id_doc INT REFERENCES Documento(id_doc) ON DELETE SET NULL,
    id_imagen INT REFERENCES Imagen(id_imagen) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS Evaluacion (
    id_evaluacion SERIAL PRIMARY KEY,
    faithfulness DOUBLE PRECISION NOT NULL CHECK (faithfulness >= 0 AND faithfulness <= 1),
    answer_relevancy DOUBLE PRECISION NOT NULL CHECK (answer_relevancy >= 0 AND answer_relevancy <= 1),
    context_recall DOUBLE PRECISION NOT NULL CHECK (context_recall >= 0 AND context_recall <= 1),
    modelo_eval VARCHAR(120) NOT NULL,
    fecha TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    id_consulta INT NOT NULL REFERENCES Consulta(id_consulta) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS Tipo_Galaxia (
    id_tipo_galaxia SERIAL PRIMARY KEY,
    nombre_tipo VARCHAR(80) NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS Tipo_Estrella (
    id_tipo_estrella SERIAL PRIMARY KEY,
    nombre_tipo VARCHAR(80) NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS Tipo_Planeta (
    id_tipo_planeta SERIAL PRIMARY KEY,
    nombre_tipo VARCHAR(80) NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS Galaxia (
    id_objeto INT PRIMARY KEY REFERENCES Objeto_Astronomico(id_objeto) ON DELETE CASCADE,
    id_tipo_galaxia INT NOT NULL REFERENCES Tipo_Galaxia(id_tipo_galaxia),
    distancia DOUBLE PRECISION CHECK (distancia IS NULL OR distancia >= 0)
);

CREATE TABLE IF NOT EXISTS Sistema_Estelar (
    id_objeto INT PRIMARY KEY REFERENCES Objeto_Astronomico(id_objeto) ON DELETE CASCADE,
    id_galaxia INT NOT NULL REFERENCES Galaxia(id_objeto) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS Estrella (
    id_objeto INT PRIMARY KEY REFERENCES Objeto_Astronomico(id_objeto) ON DELETE CASCADE,
    id_tipo_estrella INT NOT NULL REFERENCES Tipo_Estrella(id_tipo_estrella),
    id_sistema INT NOT NULL REFERENCES Sistema_Estelar(id_objeto) ON DELETE CASCADE,
    masa DOUBLE PRECISION CHECK (masa IS NULL OR masa >= 0),
    temperatura INT CHECK (temperatura IS NULL OR temperatura >= 0)
);

CREATE TABLE IF NOT EXISTS Planeta (
    id_objeto INT PRIMARY KEY REFERENCES Objeto_Astronomico(id_objeto) ON DELETE CASCADE,
    id_tipo_planeta INT NOT NULL REFERENCES Tipo_Planeta(id_tipo_planeta),
    id_sistema INT NOT NULL REFERENCES Sistema_Estelar(id_objeto) ON DELETE CASCADE,
    masa DOUBLE PRECISION CHECK (masa IS NULL OR masa >= 0),
    temperatura INT CHECK (temperatura IS NULL OR temperatura >= 0)
);

CREATE TABLE IF NOT EXISTS Luna (
    id_objeto INT PRIMARY KEY REFERENCES Objeto_Astronomico(id_objeto) ON DELETE CASCADE,
    id_planeta INT NOT NULL REFERENCES Planeta(id_objeto) ON DELETE CASCADE,
    radio DOUBLE PRECISION CHECK (radio IS NULL OR radio >= 0)
);

CREATE TABLE IF NOT EXISTS Caracteristica_Ambiental (
    id_caracteristica SERIAL PRIMARY KEY,
    id_planeta INT NOT NULL REFERENCES Planeta(id_objeto) ON DELETE CASCADE,
    tipo VARCHAR(120) NOT NULL,
    valor TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS Evaluacion_Habitabilidad (
    id_eval_habitabilidad SERIAL PRIMARY KEY,
    id_planeta INT NOT NULL REFERENCES Planeta(id_objeto) ON DELETE CASCADE,
    puntaje DOUBLE PRECISION NOT NULL CHECK (puntaje >= 0 AND puntaje <= 1),
    descripcion TEXT,
    fecha DATE NOT NULL DEFAULT CURRENT_DATE
);

CREATE TABLE IF NOT EXISTS Telescopio (
    id_telescopio SERIAL PRIMARY KEY,
    nombre VARCHAR(160) NOT NULL,
    tipo VARCHAR(120),
    ubicacion TEXT
);

CREATE TABLE IF NOT EXISTS Observacion (
    id_observacion SERIAL PRIMARY KEY,
    id_telescopio INT NOT NULL REFERENCES Telescopio(id_telescopio) ON DELETE CASCADE,
    id_objeto INT NOT NULL REFERENCES Objeto_Astronomico(id_objeto) ON DELETE CASCADE,
    fecha DATE NOT NULL DEFAULT CURRENT_DATE,
    descripcion TEXT
);
