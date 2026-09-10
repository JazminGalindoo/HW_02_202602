# Guion del video — Fases 4 y 5 (≈4 minutos)

Guion enfocado **solo en lo que hicimos nosotras**: el panel interactivo
(Fase 4) y el informe (Fase 5). No repite la adquisición ni la limpieza de
Fase 1 — para eso está `guion_video.md`, el guion completo de las 5 fases.

Preparación antes de grabar:

```bash
python run_fase4.py                 # deja los artefactos listos
streamlit run app.py
```

Ten a la mano también `report/informe.pdf` abierto en una pestaña, para las
capturas de la sección 3.

---

## 0:00 – 0:20 · Encuadre rápido

**Pantalla:** portada del panel (pestaña Resumen, sin filtrar).

> Ya tienen los datos limpios y las rutas calculadas de las fases anteriores.
> Lo que hicimos nosotras fue construirle un panel interactivo a todo eso —y
> un informe que defiende un hallazgo con números, no solo los reporta.
> Empecemos por el panel.

## 0:20 – 1:10 · El panel: indicadores y mapa

**Pantalla:** encabezado de indicadores, luego pestaña **Mapa** (coropleto).

> Arriba del todo hay cinco indicadores que se recalculan con cada filtro:
> población cubierta, población a más de 60 minutos, la mediana del tiempo,
> el peor distrito y el índice de Gini. Con los tres departamentos completos,
> el 71.8 % de la población llega en menos de media hora — pero 708 mil
> personas, un 18 %, están a más de una hora.
>
> En el mapa coloreamos cada distrito por su tiempo medio de acceso.
> *(Filtrar por departamento, uno por uno.)* Piura en verde, Ayacucho mezclado,
> y Loreto casi todo en rojo: ahí el tiempo medio pasa de 20 minutos en Piura
> a más de 700 en Loreto. Los puntos azules son los 58 hospitales con
> capacidad resolutiva — y noten que solo 23 de los 237 distritos tienen uno
> propio.

## 1:10 – 1:50 · Distribución: la media miente sola

**Pantalla:** pestaña **Distribución**.

> Este gráfico es la razón por la que no basta con dar un promedio. La media
> ponderada del tiempo de acceso es de 131 minutos. Pero la **mediana**
> ponderada —el tiempo de la persona que está justo en el medio— es de
> **2.8 minutos**. Las dos cifras son ciertas: la mayoría vive cerca de un
> hospital en la costa, y una minoría en la selva tarda días. Por eso el
> panel siempre muestra las dos, nunca una sola.

## 1:50 – 2:50 · El simulador de escenarios

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

## 2:50 – 3:10 · Brechas y calidad de datos

**Pantalla:** pestaña **Brechas**, luego **Calidad de datos**.

> El ranking de distritos críticos es descargable en CSV y ordenable por
> cualquier columna. Y la pestaña de calidad no es cosmética: muestra que el
> 23 % de los hospitales del registro nacional no tiene coordenadas, y que el
> 38 % de los centros poblados no tiene ni una vía transitable a menos de un
> kilómetro. Esas dos cifras son las que explican por qué el mapa se ve como
> se ve.

## 3:10 – 3:45 · Lo que aporta el informe

**Pantalla:** `informe.pdf`, sección de Discusión (línea recta vs. red).

> El informe no repite las tablas del panel: las usa para argumentar. Esta
> figura, por ejemplo, compara la distancia en línea recta contra la
> distancia real de la red. El factor de desvío que calibramos es 1.59, más
> alto que el 1.35 que habíamos puesto por defecto. Y lo importante no es la
> forma de la red, es su velocidad: 63 kilómetros por hora en la sierra
> contra 12.7 en Loreto. Un kilómetro en línea recta cuesta siete veces más
> tiempo en la selva que en la costa.
>
> El informe también dedica una sección completa a doce limitaciones —desde
> que no modelamos ambulancias hasta que la red vial de OpenStreetMap está
> sesgada contra lo rural— porque saber qué no puede sostener tu número es
> tan importante como el número mismo.

## 3:45 – 4:00 · Cierre

**Pantalla:** repositorio, `config.md`.

> Todo esto se reproduce con dos comandos, sin necesidad de volver a levantar
> el motor de rutas. Y ningún número está escrito a mano: panel e informe
> leen exactamente las mismas funciones, así que no pueden contradecirse.

---

## Recordatorios de grabación

- Practica el flujo de clics del simulador ANTES de grabar (seleccionar
  candidato → cambiar filtro a Loreto → seleccionar el nuevo mejor
  candidato): es la parte con más pasos y la que más impresiona si sale
  fluida.
- Si el tiempo aprieta, la sección que se puede recortar sin perder el
  argumento central es "Brechas y calidad de datos" (2:50–3:10).
- Cifras a decir en voz alta al menos una vez, porque son las que un
  evaluador va a buscar en el informe: 71.8 %, Gini 0.919, factor de desvío
  1.59, y "23 de 237 distritos".
