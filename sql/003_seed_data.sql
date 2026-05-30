-- AstroData Lab - Datos minimos de demostracion.

INSERT INTO Usuario (id_usuario, nombre, correo)
VALUES (1, 'Usuario Demo', 'demo@astrodata.local')
ON CONFLICT (correo) DO NOTHING;

INSERT INTO Tipo_Galaxia (id_tipo_galaxia, nombre_tipo)
VALUES (1, 'Espiral'), (2, 'Eliptica')
ON CONFLICT (nombre_tipo) DO NOTHING;

INSERT INTO Tipo_Estrella (id_tipo_estrella, nombre_tipo)
VALUES (1, 'G2V'), (2, 'M')
ON CONFLICT (nombre_tipo) DO NOTHING;

INSERT INTO Tipo_Planeta (id_tipo_planeta, nombre_tipo)
VALUES (1, 'Rocoso'), (2, 'Gaseoso'), (3, 'Supertierra')
ON CONFLICT (nombre_tipo) DO NOTHING;

INSERT INTO Objeto_Astronomico (id_objeto, nombre, descripcion_cientifica)
VALUES
    (1, 'Via Lactea', 'Galaxia espiral barrada que contiene el Sistema Solar.'),
    (2, 'Sistema Solar', 'Sistema estelar ubicado en el brazo de Orion de la Via Lactea.'),
    (3, 'Sol', 'Estrella tipo G2V con temperatura superficial aproximada de 5778 K.'),
    (4, 'Tierra', 'Planeta rocoso con agua liquida, atmosfera rica en nitrogeno y oxigeno, y condiciones habitables.'),
    (5, 'Luna', 'Satelite natural de la Tierra con superficie rocosa y radio aproximado de 1737 km.')
ON CONFLICT (id_objeto) DO NOTHING;

INSERT INTO Galaxia (id_objeto, id_tipo_galaxia, distancia)
VALUES (1, 1, 0)
ON CONFLICT (id_objeto) DO NOTHING;

INSERT INTO Sistema_Estelar (id_objeto, id_galaxia)
VALUES (2, 1)
ON CONFLICT (id_objeto) DO NOTHING;

INSERT INTO Estrella (id_objeto, id_tipo_estrella, id_sistema, masa, temperatura)
VALUES (3, 1, 2, 1.0, 5778)
ON CONFLICT (id_objeto) DO NOTHING;

INSERT INTO Planeta (id_objeto, id_tipo_planeta, id_sistema, masa, temperatura)
VALUES (4, 1, 2, 1.0, 288)
ON CONFLICT (id_objeto) DO NOTHING;

INSERT INTO Luna (id_objeto, id_planeta, radio)
VALUES (5, 4, 1737.4)
ON CONFLICT (id_objeto) DO NOTHING;

INSERT INTO Evaluacion_Habitabilidad (id_planeta, puntaje, descripcion)
VALUES (4, 0.98, 'Planeta con alta evidencia de habitabilidad.')
ON CONFLICT DO NOTHING;

INSERT INTO Documento (id_doc, titulo, idioma, fecha, fuente, contenido_texto, id_objeto)
VALUES (
    1,
    'Habitabilidad de la Tierra',
    'es',
    CURRENT_DATE,
    'Seed local',
    'La Tierra presenta agua liquida, atmosfera estable y temperatura media compatible con vida conocida.',
    4
)
ON CONFLICT (id_doc) DO NOTHING;

INSERT INTO Telescopio (id_telescopio, nombre, tipo, ubicacion)
VALUES (1, 'James Webb Space Telescope', 'Infrarrojo', 'Orbita L2')
ON CONFLICT (id_telescopio) DO NOTHING;

INSERT INTO Observacion (id_telescopio, id_objeto, descripcion)
VALUES (1, 4, 'Observacion demo de condiciones planetarias.')
ON CONFLICT DO NOTHING;
