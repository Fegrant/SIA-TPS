# Trabajo Práctico 3 - Perceptrón Simple y Multicapa

En este TP se explora la implementacion perceptrones simples lineales, no lineales. Y de perceptrones multicapa.

## Dependencias

- Python **>= 3.11**
- Pipenv

## Set up

Primero se deben descargar las dependencias a usar en el programa. Para ello podemos hacer uso de los archivos _Pipfile_ y _Pipfile.lock_ provistos, que ya las tienen detalladas. Para usarlos se debe correr en la carpeta del TP1:

```bash
$> pipenv shell
$> pipenv install
```

Esto creará un nuevo entorno virtual, en el que se instalarán las dependencias a usar, que luego se borrarán una vez se cierre el entorno.

**NOTA:** Previo a la instalación se debe tener descargado **python** y **pipenv**, pero se omite dicho paso en esta instalación.

## Cómo Correr

python eji.py
donde i es el numero de ejercicio a ejecutar

## Archivo de Configuración:

### Configuraciones Basicas

**Nota: opción_a | opción_b | opción_c representa un parámetro que puede tomar únicamente esas opciones**

```json5
{
    "perceptron": {
        "simple":{
            "number_of_inputs": 2,
            "epochs" : 1000,
            "learning_rate" : 0.1,
            "accepted_error": 0.05

        },
        "lineal": {
            "number_of_inputs": 3,
            "epochs" : 1000,
            "learning_rate" : 0.005,
            "accepted_error": 0.05

        },
        "no-lineal": {
            "number_of_inputs": 3,
            "epochs" : 100000,
            "learning_rate" : 0.01,
            "beta": 0.1,
            "accepted_error": 0.001,
            "activation_function": 2
        },
        "multilayer": {
            "number_of_inputs": 2,
            "hidden_layers": [10],
            "number_of_outputs": 1,
            "epochs" : 10000,
            "learning_rate" : 0.01,
            "beta": 0.1,
            "convergence_threshold": 0.005,
            "momentum": 0.9
        }
    },
    "ex2_test_size": 0.2,
    "ex3_test_size": 0.25,
    "ex3c_noise": 0.5
}
```

### Archivos de salida

Los archivos de salida son estadísticas en formato `.csv`, que se encuentran en la carpeta `results`.
