# Simulación de Lente Gravitacional Interactiva

Este proyecto es una simulación interactiva de una lente gravitacional utilizando un modelo de Esfera Isoterma Singular (SIS) con Shear externo. Permite visualizar cómo la gravedad de una lente (como un agujero negro o una galaxia) distorsiona la luz de una galaxia de fondo.

## Requisitos

Para ejecutar este proyecto, necesitas tener instalado Python y los siguientes paquetes:

- `numpy`
- `matplotlib`

## Instalación

Puedes instalar las dependencias necesarias ejecutando el siguiente comando en tu terminal:

```bash
pip install numpy matplotlib
```

## Uso

Para correr la simulación, asegúrate de estar en el directorio del proyecto y ejecuta el script principal:

```bash
python lente_gravitacion_interactivo.py
```

## Controles Interactivos

Una vez que se abra la ventana de la simulación, podrás usar los controles deslizantes para ajustar los parámetros en tiempo real:

- **Lente Principal**:
    - `Theta_E`: Radio de Einstein (proporcional a la masa de la lente).
    - `Gamma`: Magnitud del Shear externo (efecto de masas cercanas).
    - `PA_gamma`: Ángulo del Shear.

- **Fuente (Galaxia de fondo)**:
    - `beta_x`, `beta_y`: Posición de la fuente.
    - `Re`: Radio efectivo (tamaño).
    - `n_Sersic`: Perfil de brillo.
    - `q`: Elipticidad (relación axial).
    - `PA_src`: Ángulo de rotación.

- **Visualización**:
    - `Ruido`: Agrega ruido simulado a la imagen.
    - `Ver Curva Crítica`: Muestra u oculta la línea naranja que representa el radio de Einstein.
    - `Resetear`: Vuelve todos los valores a su configuración inicial.

