import os
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    Input
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

plt.rcParams['figure.figsize'] = [10, 6]
os.makedirs('export', exist_ok=True)

info_directorio = Path('./ImagenTrainning/imagenes') 
if not info_directorio.exists():
    raise FileNotFoundError(f"Directorio {info_directorio} no encontrado")

subdirectorios = [p.name for p in info_directorio.iterdir() if p.is_dir()]
print(f"Subdirectorios encontrados: {subdirectorios}")

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 80
VAL_SPLIT = 0.2

entrenamiento_datos = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest',
    validation_split=VAL_SPLIT
)

validacion_datos = ImageDataGenerator(
    rescale=1./255,
    validation_split=VAL_SPLIT
)

entrenamiento = entrenamiento_datos.flow_from_directory(
    info_directorio,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=SEED
)

validacion = validacion_datos.flow_from_directory(
    info_directorio,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=SEED
)

print(f"Informacion de entrenamiento: {next(entrenamiento)[0].shape}")


CANTIDAD_CLASES = len(entrenamiento.class_indices)
LISTA_NOMBRE_CLASES = [None] * CANTIDAD_CLASES
for nombre, idx in entrenamiento.class_indices.items():
    LISTA_NOMBRE_CLASES[idx] = nombre

print(f"Clases indices (mapa): {entrenamiento.class_indices}")
print(f"Numero de clases: {CANTIDAD_CLASES}")
print(f"Lista de nombres por indice: {LISTA_NOMBRE_CLASES}")

modelo = Sequential([
    Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(CANTIDAD_CLASES, activation='softmax')
])


modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

modelo.summary()

balanceo = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(entrenamiento.classes),
    y=entrenamiento.classes
)
balanceo_clases = dict(enumerate(balanceo))
print(f"Pesos de clase para balanceo: {balanceo_clases}")


entrenar_modelo = modelo.fit(
    entrenamiento,
    epochs=EPOCHS,
    validation_data=validacion,
    class_weight=balanceo_clases,
    verbose=1
)

print("\nEvaluando modelo...")
resultados = modelo.evaluate(validacion, verbose=1)

if len(resultados) >= 4:
    loss, accuracy, precision, recall = resultados[:4]
    print(f'\nMetricas (evaluate):')
    print(f'Perdida: {loss:.4f}')
    print(f'Exactitud: {accuracy*100:.2f}%')
    print(f'Precision: {precision*100:.2f}%')
    print(f'Recall: {recall*100:.2f}%')
else:
    print("Resultados evaluate:", resultados)

print("\nGenerando matriz de confusion...")
Y_true = validacion.classes
Y_pred_probs = modelo.predict(validacion, verbose=1)
Y_pred = np.argmax(Y_pred_probs, axis=1)

cm = confusion_matrix(Y_true, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LISTA_NOMBRE_CLASES)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusion")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('export/matriz_confusion.png', dpi=300, bbox_inches='tight')
plt.show()


print("\nReporte de Clasificacion:")
print(classification_report(Y_true, Y_pred, target_names=LISTA_NOMBRE_CLASES, digits=4))


plt.plot(entrenar_modelo.history['loss'], label='Train Loss')
plt.plot(entrenar_modelo.history['val_loss'], label='Validation Loss')
plt.title('Perdida durante el entrenamiento')
plt.xlabel('epoca')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('export/curva_aprendizaje.png', dpi=300, bbox_inches='tight')
plt.show()

def predecir_imagen(modelo, ruta_imagen, nombres_clases):
    img = load_img(ruta_imagen, target_size=IMG_SIZE)
    img_arreglo = img_to_array(img) / 255.0
    img_arreglo = np.expand_dims(img_arreglo, axis=0)

    probabilidades = modelo.predict(img_arreglo, verbose=0)[0]
    clase_index = int(np.argmax(probabilidades))
    clase_etiqueta = nombres_clases[clase_index]
    probabilidad = float(probabilidades[clase_index])

    return {
        'clase': clase_etiqueta,
        'clase_index': clase_index,
        'probabilidad': probabilidad
    }

print("\nRealizando predicciones de prueba...")
imagenes_prueba = [
    './Test_Images/banana_test.webp',
    './Test_Images/frijol_Test.png',
    './Test_Images/garbanzos_test.webp',
    "./Test_Images/leche_test.webp",
    "./Test_Images/mandarina_test.png",
    "./Test_Images/manzana_Test.jpg",
    "./Test_Images/queso_tajado_test.png",
    "./Test_Images/yogurt_Test.jpeg"
]

for imagen_ruta in imagenes_prueba:
    if os.path.exists(imagen_ruta):
        prediccion = predecir_imagen(modelo, imagen_ruta, LISTA_NOMBRE_CLASES)
        print(f"\nPrediccion para {imagen_ruta}:")
        print(f"Clase: {prediccion['clase']} (Indice: {prediccion['clase_index']})")
        print(f"Probabilidad: {prediccion['probabilidad']:.4f}")
    else:
        print(f"\nArchivo {imagen_ruta} no encontrado")


modelo.save("export/modelo_FrutasLacteosGarbanzos.h5")
with open("export/labels.txt", "w", encoding="utf-8") as f:
    for name in LISTA_NOMBRE_CLASES:
        f.write(name + "\n")

print("\nModelo guardado como 'export/FrutasLacteosGarbanzos.h5' y labels en export/labels.txt")
print("\n¡Entrenamiento completado! SISAS")