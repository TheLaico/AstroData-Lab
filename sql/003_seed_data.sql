-- AstroData Lab - Datos minimos de demostracion.
-- Compatible con bases existentes aunque no tengan UNIQUE en columnas naturales.

INSERT INTO Usuario (nombre, correo)
SELECT 'Usuario Demo', 'demo@astrodata.local'
WHERE NOT EXISTS (
    SELECT 1 FROM Usuario WHERE correo = 'demo@astrodata.local'
);

INSERT INTO Tipo_Galaxia (nombre_tipo)
SELECT 'Espiral'
WHERE NOT EXISTS (SELECT 1 FROM Tipo_Galaxia WHERE nombre_tipo = 'Espiral');

INSERT INTO Tipo_Galaxia (nombre_tipo)
SELECT 'Eliptica'
WHERE NOT EXISTS (SELECT 1 FROM Tipo_Galaxia WHERE nombre_tipo = 'Eliptica');

INSERT INTO Tipo_Estrella (nombre_tipo)
SELECT 'G2V'
WHERE NOT EXISTS (SELECT 1 FROM Tipo_Estrella WHERE nombre_tipo = 'G2V');

INSERT INTO Tipo_Estrella (nombre_tipo)
SELECT 'M'
WHERE NOT EXISTS (SELECT 1 FROM Tipo_Estrella WHERE nombre_tipo = 'M');

INSERT INTO Tipo_Planeta (nombre_tipo)
SELECT 'Rocoso'
WHERE NOT EXISTS (SELECT 1 FROM Tipo_Planeta WHERE nombre_tipo = 'Rocoso');

INSERT INTO Tipo_Planeta (nombre_tipo)
SELECT 'Gaseoso'
WHERE NOT EXISTS (SELECT 1 FROM Tipo_Planeta WHERE nombre_tipo = 'Gaseoso');

INSERT INTO Tipo_Planeta (nombre_tipo)
SELECT 'Supertierra'
WHERE NOT EXISTS (SELECT 1 FROM Tipo_Planeta WHERE nombre_tipo = 'Supertierra');

INSERT INTO Objeto_Astronomico (nombre, descripcion_cientifica)
SELECT 'Via Lactea', 'Galaxia espiral barrada que contiene el Sistema Solar.'
WHERE NOT EXISTS (SELECT 1 FROM Objeto_Astronomico WHERE nombre = 'Via Lactea');

INSERT INTO Objeto_Astronomico (nombre, descripcion_cientifica)
SELECT 'Sistema Solar', 'Sistema estelar ubicado en el brazo de Orion de la Via Lactea.'
WHERE NOT EXISTS (SELECT 1 FROM Objeto_Astronomico WHERE nombre = 'Sistema Solar');

INSERT INTO Objeto_Astronomico (nombre, descripcion_cientifica)
SELECT 'Sol', 'Estrella tipo G2V con temperatura superficial aproximada de 5778 K.'
WHERE NOT EXISTS (SELECT 1 FROM Objeto_Astronomico WHERE nombre = 'Sol');

INSERT INTO Objeto_Astronomico (nombre, descripcion_cientifica)
SELECT 'Tierra', 'Planeta rocoso con agua liquida, atmosfera rica en nitrogeno y oxigeno, y condiciones habitables.'
WHERE NOT EXISTS (SELECT 1 FROM Objeto_Astronomico WHERE nombre = 'Tierra');

INSERT INTO Objeto_Astronomico (nombre, descripcion_cientifica)
SELECT 'Luna', 'Satelite natural de la Tierra con superficie rocosa y radio aproximado de 1737 km.'
WHERE NOT EXISTS (SELECT 1 FROM Objeto_Astronomico WHERE nombre = 'Luna');

INSERT INTO Galaxia (id_objeto, id_tipo_galaxia, distancia)
SELECT o.id_objeto, tg.id_tipo_galaxia, 0
FROM Objeto_Astronomico o
JOIN Tipo_Galaxia tg ON tg.nombre_tipo = 'Espiral'
WHERE o.nombre = 'Via Lactea'
AND NOT EXISTS (SELECT 1 FROM Galaxia g WHERE g.id_objeto = o.id_objeto);

INSERT INTO Sistema_Estelar (id_objeto, id_galaxia)
SELECT sistema.id_objeto, galaxia.id_objeto
FROM Objeto_Astronomico sistema
JOIN Objeto_Astronomico galaxia ON galaxia.nombre = 'Via Lactea'
WHERE sistema.nombre = 'Sistema Solar'
AND NOT EXISTS (
    SELECT 1 FROM Sistema_Estelar se WHERE se.id_objeto = sistema.id_objeto
);

INSERT INTO Estrella (id_objeto, id_tipo_estrella, id_sistema, masa, temperatura)
SELECT sol.id_objeto, te.id_tipo_estrella, sistema.id_objeto, 1.0, 5778
FROM Objeto_Astronomico sol
JOIN Tipo_Estrella te ON te.nombre_tipo = 'G2V'
JOIN Objeto_Astronomico sistema ON sistema.nombre = 'Sistema Solar'
WHERE sol.nombre = 'Sol'
AND NOT EXISTS (SELECT 1 FROM Estrella e WHERE e.id_objeto = sol.id_objeto);

INSERT INTO Planeta (id_objeto, id_tipo_planeta, id_sistema, masa, temperatura)
SELECT tierra.id_objeto, tp.id_tipo_planeta, sistema.id_objeto, 1.0, 288
FROM Objeto_Astronomico tierra
JOIN Tipo_Planeta tp ON tp.nombre_tipo = 'Rocoso'
JOIN Objeto_Astronomico sistema ON sistema.nombre = 'Sistema Solar'
WHERE tierra.nombre = 'Tierra'
AND NOT EXISTS (SELECT 1 FROM Planeta p WHERE p.id_objeto = tierra.id_objeto);

INSERT INTO Luna (id_objeto, id_planeta, radio)
SELECT luna.id_objeto, tierra.id_objeto, 1737.4
FROM Objeto_Astronomico luna
JOIN Objeto_Astronomico tierra ON tierra.nombre = 'Tierra'
WHERE luna.nombre = 'Luna'
AND NOT EXISTS (SELECT 1 FROM Luna l WHERE l.id_objeto = luna.id_objeto);

INSERT INTO Evaluacion_Habitabilidad (id_planeta, puntaje, descripcion)
SELECT tierra.id_objeto, 0.98, 'Planeta con alta evidencia de habitabilidad.'
FROM Objeto_Astronomico tierra
WHERE tierra.nombre = 'Tierra'
AND NOT EXISTS (
    SELECT 1 FROM Evaluacion_Habitabilidad eh WHERE eh.id_planeta = tierra.id_objeto
);

INSERT INTO Documento (titulo, idioma, fecha, fuente, contenido_texto, id_objeto)
SELECT
    'Habitabilidad de la Tierra',
    'es',
    CURRENT_DATE,
    'Seed local',
    'La Tierra presenta agua liquida, atmosfera estable y temperatura media compatible con vida conocida.',
    tierra.id_objeto
FROM Objeto_Astronomico tierra
WHERE tierra.nombre = 'Tierra'
AND NOT EXISTS (SELECT 1 FROM Documento WHERE titulo = 'Habitabilidad de la Tierra');

INSERT INTO Telescopio (nombre, tipo, ubicacion)
SELECT 'James Webb Space Telescope', 'Infrarrojo', 'Orbita L2'
WHERE NOT EXISTS (
    SELECT 1 FROM Telescopio WHERE nombre = 'James Webb Space Telescope'
);

INSERT INTO Observacion (id_telescopio, id_objeto, fecha, descripcion)
SELECT t.id_telescopio, o.id_objeto, CURRENT_DATE, 'Observacion demo de condiciones planetarias.'
FROM Telescopio t
JOIN Objeto_Astronomico o ON o.nombre = 'Tierra'
WHERE t.nombre = 'James Webb Space Telescope'
AND NOT EXISTS (
    SELECT 1
    FROM Observacion obs
    WHERE obs.id_telescopio = t.id_telescopio
    AND obs.id_objeto = o.id_objeto
    AND obs.descripcion = 'Observacion demo de condiciones planetarias.'
);
