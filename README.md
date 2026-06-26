# Agentic Memory Layer

Una capa de memoria estructurada y persistente para agentes autonomos de Inteligencia Artificial. Este modulo resuelve el problema de la "amnesia" en LLMs dividiendo la memoria en un buffer conversacional de corto plazo con sintesis automatica y un almacen episódico de largo plazo que simula la curva del olvido de Ebbinghaus, aplicando decaimiento exponencial-temporal de la relevancia de los recuerdos.

## Arquitectura de Memoria Hibrida

El sistema implementa dos almacenes acoplados:

### 1. Memoria de Corto Plazo (Conversational Buffer)
*   Mantiene una lista de los ultimos N turnos del dialogo activo.
*   Al sobrepasar el limite (por defecto 5 interacciones dobles), se dispara de forma automatica un algoritmo de sintesis que resume las interacciones antiguas y condensa el contexto, liberando espacio en la ventana de atencion del agente sin perder el hilo historico.

### 2. Memoria de Largo Plazo (Episodic Decay Memory)
*   **Formula de Olvido Exponencial:** Para cada consulta de recuerdo, se puntuan los hechos guardados utilizando la formula:
    $$Score = Similitud\_Coseno \cdot Importancia \cdot e^{-\lambda \cdot \Delta t}$$
    Donde $\Delta t$ es el tiempo transcurrido, e $\lambda$ es el coeficiente de decaimiento (decay_factor). Hechos antiguos pierden peso progresivamente a menos que sean altamente relevantes o tengan una gran importancia original.
*   **Integracion con NanoVectorDB:** El modulo intenta importar dinamicamente la base de datos vectorial local `NanoVectorDB` del repositorio hermano. Si esta disponible, indexa y busca los recuerdos vectorialmente en disco.
*   **Contingencia Standalone:** En caso de estar offline o sin el modulo hermano, calcula similitudes vectoriales deterministicas en base a hashes de caracteres (vectores unitarios locales) permitiendo su ejecucion de forma aislada e inmediata.

## Requisitos e Instalacion

*   Python 3.8 o superior
*   NumPy

Para instalar las dependencias locales, ejecute:

```bash
pip install -r requirements.txt
```

## Pruebas y Verificacion

1.  **Ejecutar Suite de Pruebas Unitarias:**
    ```bash
    python -m unittest test_memory.py
    ```
2.  **Ejecutar Demostracion de Curva de Olvido:**
    ```bash
    python example.py
    ```
    El script guardara varios hechos sobre el usuario, realizara busquedas conceptuales inmediatas y luego simulara lapsos de tiempo (30 segundos y 10 minutos) para ilustrar como decaen los scores de recuerdo y que porcentaje de retencion se mantiene en el agente.
