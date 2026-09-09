# Guion del video de presentación (5 minutos)

Guion de apoyo para grabar la demostración. Cada bloque indica **qué se ve en
pantalla** y **qué se dice**. Los números están tomados de `data/outputs/`; si
se vuelve a correr el pipeline con datos actualizados, hay que revisarlos.

Preparación antes de grabar:

```bash
python run_fase4.py                 # deja los artefactos listos
streamlit run app/dashboard.py      # abrir en el navegador, pestaña "Resumen"
```

---

## 0:00 – 0:35 · El problema

**Pantalla:** portada del informe o la figura del mapa (`fig_mapa_acceso.png`).

> En una emergencia obstétrica o un accidente grave, lo que decide el
> desenlace no es tener una posta cerca: es tener cerca un establecimiento que
> pueda **resolver**. Medimos cuánto tarda la población de Piura, Ayacucho y
> Loreto en llegar por carretera al hospital resolutivo más cercano —categoría
> II-1 o superior—, y qué tan desigual es ese tiempo. Elegimos esos tres
> departamentos porque son costa, sierra y selva.

## 0:35 – 1:20 · De dónde salen los datos, y qué falló

**Pantalla:** pestaña **Calidad de datos** del panel, bloque de adquisición.

> Cuatro fuentes: RENIPRESS para la oferta, centros poblados de SIGMED para la
> demanda, el Censo 2017 del INEI para la población y OpenStreetMap para la red
> vial. Dos de ellas no se pueden descargar de forma programática: el portal de
> datos abiertos responde con un error a cualquier cliente que no sea un
> navegador, y SIGMED no tiene API. Lo dejamos registrado en el log y lo
> reportamos: es un hallazgo sobre el estado de los datos abiertos peruanos, no
> algo que convenga esconder.

## 1:20 – 2:10 · Cómo se limpió

**Pantalla:** misma pestaña, tabla de reglas de calidad.

> Seis reglas, una fila por regla: cuántos registros se marcaron, qué se hizo y
> por qué. Ninguna descarta en silencio; cada fila marcada conserva su bandera
> en los datos procesados. El dato grueso está arriba: el **23 % de los
> establecimientos del registro nacional no tiene coordenadas**. Sin coordenada
> no hay ruta posible, así que quedan fuera del cálculo, con la bandera puesta.
> Y abajo, el enganche a la red vial: el **38 % de los centros poblados no
> tiene una vía transitable en auto a menos de un kilómetro**, mientras que el
> 100 % de los hospitales sí la tiene. Los hospitales están donde están las
> carreteras.

## 2:10 – 3:10 · El resultado principal

**Pantalla:** pestaña **Resumen**, luego **Mapa** (coropleto).

> El 71.8 % de la población llega en menos de 30 minutos. Ese número, solo,
> engaña. *(Filtrar por departamento en la barra lateral, uno por uno.)* En
> Piura es 81 %; en Ayacucho, 35 %; en Loreto, 50 %, con un 32 % adicional a
> más de dos horas y un 8 % sin ninguna ruta. El tiempo medio ponderado pasa de
> 20 minutos en Piura a casi doce horas en Loreto.
>
> Fíjense en algo del panel: al mover el filtro **no se muestra una tabla
> precalculada**, se vuelven a llamar las mismas funciones de métricas que
> produjeron el informe. Panel e informe no pueden contradecirse, y hay un test
> que lo verifica.

## 3:10 – 3:50 · Desigualdad y modo de viaje

**Pantalla:** pestaña **Equidad** (curva de Lorenz), luego **Modos de viaje**.

> El Gini del tiempo de acceso es 0.919. No significa ``mal acceso promedio'':
> significa que el tiempo total de viaje está concentrado en una minoría
> identificable, rural y amazónica. La brecha rural/urbana es de 2.3 veces.
>
> Y el modo importa: a pie, la mediana es **once veces** el tiempo en auto, y
> en un tercio de los centros poblados urbanos **cambia cuál es el
> establecimiento más cercano** según cómo se viaje. Un indicador de
> accesibilidad que supone auto disponible describe a quien tiene auto.

## 3:50 – 4:25 · La limitación que define el trabajo

**Pantalla:** pestaña **Mapa**, Loreto; o `fig_ranking_criticos.png`.

> Los quince distritos con peor acceso están todos en Loreto, con medias de más
> de ochenta horas. Ese número no describe un viaje real: describe lo que
> costaría hacerlo por carretera. En Loreto se viaja por río, y los ríos no
> están en la red vial de OpenStreetMap. Nuestro modelo **no puede medir el
> acceso real amazónico**, y decirlo con claridad es más útil que dar un número
> verosímil pero falso. La recomendación que sale de ahí es concreta: antes de
> más modelamiento vial, hace falta producir el dato que no existe, una red
> fluvial navegable con velocidades por tipo de embarcación.

## 4:25 – 5:00 · Reproducibilidad y cierre

**Pantalla:** el repositorio, `config.md`, y la salida de `pytest`.

> Todo se reproduce con cinco comandos. Ningún parámetro está escrito dentro
> del código: departamentos, categorías resolutivas, umbrales, semilla del
> muestreo y puertos de OSRM viven en `config.md`, que es a la vez la
> documentación y la fuente que lee el código. Cambiar de departamento no
> requiere tocar un solo `.py`. La suite de pruebas cubre las reglas de
> validación, las métricas y el contrato entre el panel y el módulo de
> métricas.
>
> Conclusión: solo 23 de los 237 distritos tienen un establecimiento resolutivo
> propio. El problema no es la cantidad de establecimientos, es su capacidad y
> su distribución.

---

## Recordatorios de grabación

- Mostrar el panel **en vivo**, moviendo un filtro de verdad: es lo que
  distingue un panel de una captura de pantalla.
- Decir en voz alta, al menos una vez, sobre qué universo se habla: la muestra
  enrutada de 5,002 centros poblados de 19,460, ponderada por población.
- No leer las tablas número por número; señalar el contraste y seguir.
