# Entrenamiento de Modelo GAN para Generación de Imágenes MNIST

Este proyecto implementa un modelo GAN (Generative Adversarial Network) para generar imágenes similares a las del conjunto de datos MNIST utilizando TensorFlow y Keras.

## Descripción

El proyecto consiste en entrenar un modelo GAN para generar imágenes de dígitos escritos a mano. El modelo se entrena utilizando el conjunto de datos MNIST y consta de dos partes principales: un generador y un discriminador.

## Requisitos

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn

Puedes instalar las dependencias necesarias utilizando el siguiente comando:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Configuración

El proyecto utiliza una clase `Config` para definir los hiperparámetros del modelo:

```python
class Config:
    BUFFER_SIZE = 60000
    BATCH_SIZE = 128
    EPOCHS = 120
    NOISE_DIM = 100
    NUM_EXAMPLES_TO_GENERATE = 16
    LEARNING_RATE = 2e-4
    BETA_1 = 0.5
    LAMBDA = 10
```

## Entrenamiento del Modelo

El entrenamiento del modelo se realiza en dos fases:

1. **Entrenamiento inicial hasta la época 200**: Se entrena el modelo desde cero hasta la época 200 pero se detuvo por que se termino el tiempo de GPU brindada por Google Colab.
2. **Entrenamiento adicional hasta la época 120**: Se continúa el entrenamiento desde el último checkpoint guardado hasta la época 120.

### Función de Entrenamiento

La función principal de entrenamiento es `train(dataset, epochs, start_epoch=0)`, que realiza lo siguiente:

- Carga los datos de entrenamiento.
- Entrena el modelo GAN durante el número especificado de épocas.
- Guarda checkpoints cada 10 épocas.
- Genera y guarda imágenes de ejemplo durante el entrenamiento.

### Ejemplo de Uso

Para entrenar el modelo, simplemente ejecuta el script principal:

```bash
python gan_model.py
```

## Evaluación del Modelo

Después de entrenar el modelo, se pueden evaluar las métricas del discriminador y generar nuevas imágenes utilizando el generador.

### Métricas del Discriminador

Las métricas del discriminador se calculan utilizando el conjunto de datos de prueba MNIST:

```python
accuracy, precision, recall, f1_score = evaluate_discriminator(discriminator, test_images, test_labels)
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1_score:.4f}')
```

### Generación de Imágenes

Para generar nuevas imágenes utilizando el generador:

```python
generate_and_evaluate_images(generator, num_images=16)
```

## Resultados

### Métricas del Discriminador

| Métrica | Valor |
|---------|-------|
| Accuracy | 0.5928 |
| Precision | 0.6018 |
| Recall | 0.6018 |
| F1 Score | 0.6018 |

## Conclusión

Este proyecto demuestra cómo entrenar un modelo GAN para generar imágenes de dígitos escritos a mano utilizando TensorFlow y Keras. El modelo se entrena en dos fases y se evalúa utilizando métricas estándar.

## Contacto

Si tienes alguna pregunta o comentario, no dudes en contactarme.
