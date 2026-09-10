# Guion del video — Fases 4 y 5 (≈5:40)

Guion enfocado **solo en lo que hicimos nosotras**: el panel interactivo
(Fase 4) y el informe (Fase 5). No repite la adquisición ni la limpieza de
Fase 1 — para eso está `guion_video.md`, el guion completo de las 5 fases.

La primera versión de este guion duraba 4 minutos y le daba a la Fase 5 solo
35 segundos. Esta versión le da a cada fase su peso real: **~2:50 al panel**
y **~2:30 al informe**, más encuadre y cierre.

Preparación antes de grabar:

```bash
python run_fase4.py                 # deja los artefactos listos
streamlit run app.py
```

Ten también `report/informe.pdf` abierto en otra pestaña/ventana, para las
capturas de la parte 2.

---

# PARTE 1 — Panel interactivo (Fase 4) · 0:00 – 2:50

## 0:00 – 0:15 · Encuadre rápido

**Pantalla:** portada del panel (pestaña Resumen, sin filtrar).

> Ya tienen los datos limpios y las rutas calculadas de las fases anteriores.
> Lo que hicimos nosotras fue construirle un panel interactivo a todo eso —y
> un informe que defiende un hallazgo con números, no solo los reporta.
> Empecemos por el panel.

## 0:15 – 1:00 · Indicadores y mapa

**Pantalla:** encabezado de indicadores, luego pestaña **Mapa** (coropleto).

> Arriba del todo hay cinco indicadores que se recalculan con cada filtro:
> población cubierta, población a más de 60 minutos, la mediana del tiempo,
> el peor distrito y el índice de Gini. Con los tres departamentos completos,
> el 71.8 % de la población llega en menos de media hora — pero 708 mil
> personas, un 18 %, están a más de una hora.
>
> En el mapa coloreamos cada distrito por su tiempo medio de acceso.
> *(Filtrar por departamento, uno por uno.)* Piura en verde, Ayacucho
> mezclado, y Loreto casi todo en rojo: ahí el tiempo medio pasa de 20
> minutos en Piura a más de 700 en Loreto. Los puntos azules son los 58
> hospitales con capacidad resolutiva — y noten que solo 23 de los 237
> distritos tienen uno propio.

## 1:00 – 1:35 · Distribución: la media miente sola

**Pantalla:** pestaña **Distribución**.

> Este gráfico es la razón por la que no basta con dar un promedio. La media
> ponderada del tiempo de acceso es de 131 minutos. Pero la **mediana**
> ponderada —el tiempo de la persona que está justo en el medio— es de
> **2.8 minutos**. Las dos cifras son ciertas: la mayoría vive cerca de un
> hospital en la costa, y una minoría en la selva tarda días. Por eso el
> panel siempre muestra las dos, nunca una sola.

## 1:35 – 2:30 · El simulador de escenarios

**Pantalla:** pestaña **Simulador de escenarios**.

> Esta es la pieza nueva más importante: convertir el diagnóstico en una
> decisión. Se elige un establecimiento que hoy es I-3 o I-4 y el panel
> recalcula qué pasaría si tuviera capacidad resolutiva. *(Seleccionar el
> primero del ranking.)* Ascender el establecimiento de Bernal, en Sechura,
> acercaría a 141 mil personas a menos de media hora: la cobertura sube de
> 71.8 a 75.4 por ciento.
>
> Miren este detalle: los cinco mejores candidatos individuales suman 423 mil
> personas beneficiadas, pero elegidos juntos dan otra vez 141 mil. Se
> solapan, cubren a la misma gente. Por eso el panel calcula el escenario
> combinado en vez de sumar un ranking.
>
> Y si filtro solo Loreto *(cambiar filtro)*, la respuesta cambia por
> completo: el mejor candidato ya no está en la costa, está en Caballococha,
> en Ramón Castilla —el distrito con peor acceso de todo el estudio—. A quién
> priorizar es una decisión política, y el panel la hace explícita en vez de
> esconderla en un promedio nacional.

## 2:30 – 2:50 · Brechas y calidad de datos

**Pantalla:** pestaña **Brechas**, luego **Calidad de datos**.

> El ranking de distritos críticos es descargable en CSV y ordenable por
> cualquier columna. Y la pestaña de calidad no es cosmética: muestra que el
> 23 % de los hospitales del registro nacional no tiene coordenadas, y que el
> 38 % de los centros poblados no tiene ni una vía transitable a menos de un
> kilómetro. Esas dos cifras explican por qué el mapa se ve como se ve.

---

# PARTE 2 — Informe (Fase 5) · 2:50 – 5:20

## 2:50 – 3:15 · Qué es el informe y de dónde salen los datos

**Pantalla:** `informe.pdf`, portada + tabla de fuentes (sección 2).

> Todo lo que acaban de ver en el panel también está en un informe de doce
> páginas, con las figuras y tablas generadas directamente por el código
> —nada pegado a mano ni capturado de pantalla—. Empieza documentando de
> dónde sale cada dato, con fecha de acceso y licencia. Y ahí ya hay un
> hallazgo: dos de las cinco fuentes oficiales no se pueden descargar de
> forma automática. El portal de RENIPRESS bloquea a cualquier cliente que no
> sea un navegador, y SIGMED, la fuente de los centros poblados, ni siquiera
> declara una licencia de uso.

## 3:15 – 3:45 · Metodología y resultados citables

**Pantalla:** sección de Metodología y Resultados, con las tablas
`informe_departamentos` y `informe_ranking` visibles.

> La metodología explica, para que cualquiera pueda reproducir nuestros
> números: qué categoría cuenta como resolutiva, cómo se calculó el tiempo
> con OSRM, y cómo se ponderó cada punto por su población censada. Y los
> resultados no repiten las tablas del panel tal cual: las presentan como
> evidencia. Por ejemplo, la tabla de tiempo medio por departamento es la
> misma que vieron en el mapa, pero aquí se usa para argumentar que Piura,
> Ayacucho y Loreto son, en la práctica, tres problemas distintos.

## 3:45 – 4:20 · Discusión: línea recta contra red

**Pantalla:** figura de línea recta vs. red (sección de Discusión).

> Esta es la figura que más nos costó y la que más dice. Comparamos, para
> cada par origen-destino que sí se pudo enrutar, la distancia en línea recta
> contra la distancia real por la carretera. El factor de desvío que
> calibramos es 1.59, más alto que el 1.35 que habíamos puesto por defecto
> antes de medirlo. Pero lo importante no es la forma de la red, es su
> velocidad: 63 kilómetros por hora en la sierra y la costa, contra 12.7 en
> Loreto. Combinando las dos cosas, un kilómetro en línea recta cuesta siete
> veces más tiempo en la selva que en la costa. Por eso ninguna estimación
> por distancia, con o sin corrección, sirve en la Amazonía.

## 4:20 – 4:55 · Limitaciones: lo que el número no puede sostener

**Pantalla:** sección de Limitaciones (lista enumerada).

> El informe dedica una sección completa a doce limitaciones, ordenadas por
> qué tanto pueden cambiar la conclusión. Las cuatro que más pesan: primero,
> el modelo no representa el transporte fluvial, que es como en realidad se
> mueve la gente en Loreto. Segundo, OpenStreetMap está mejor mapeado en zonas
> urbanas que rurales, así que parte de esa brecha rural puede ser un sesgo
> del mapa y no del territorio. Tercero, no modelamos si existe una ambulancia
> disponible: todos los tiempos suponen un vehículo listo en el momento del
> evento. Y cuarto, el 23 % de los hospitales del registro no tiene
> coordenadas, así que quedaron fuera del cálculo. Ponerlas por escrito, con
> la dirección del sesgo cuando se puede saber, es lo que distingue un número
> honesto de uno solo verosímil.

## 4:55 – 5:20 · Conclusiones y recomendaciones

**Pantalla:** sección final del informe.

> Cerramos con seis conclusiones concretas. La política relevante no es
> construir más postas, es elevar la categoría de establecimientos ya
> existentes en nodos bien conectados —el simulador que vimos hace un
> momento sirve exactamente para eso—. Para Loreto, la recomendación es
> metodológica: antes de seguir modelando carreteras, hace falta un dato que
> hoy no existe, una red fluvial navegable. Y para las fuentes de datos: si
> RENIPRESS publicara coordenadas completas y SIGMED expusiera su capa con
> una API, un cuarto de la oferta de salud dejaría de ser invisible para
> cualquier análisis geográfico.

---

## 5:20 – 5:40 · Cierre

**Pantalla:** repositorio, `config.md`, salida de `pytest`.

> Todo esto se reproduce con dos comandos, sin volver a levantar el motor de
> rutas. Y ningún número está escrito a mano: el panel y el informe leen
> exactamente las mismas funciones, así que no pueden contradecirse.

---

## Recordatorios de grabación

- Practica el flujo de clics del simulador ANTES de grabar (seleccionar
  candidato → cambiar filtro a Loreto → seleccionar el nuevo mejor
  candidato): es la parte con más pasos y la que más impresiona si sale
  fluida.
- Si el tiempo aprieta, recorta primero "Metodología y resultados citables"
  (3:15–3:45): es la parte más reemplazable por texto en pantalla en vez de
  narración.
- Cifras a decir en voz alta al menos una vez, porque son las que un
  evaluador va a buscar en el informe: 71.8 %, Gini 0.919, factor de desvío
  1.59, "23 de 237 distritos" y "doce limitaciones".
- El total da **5:40**. Si te piden estrictamente 4 minutos, quita del todo
  la Parte 2 y usa en su lugar solo el bloque "Discusión: línea recta contra
  red" (3:45–4:20) — es el argumento más fuerte del informe y cabe solo.
