# De la interpolación clásica a la Super-Resolución con Deep Learning y Transformers

> Documentación técnica complementaria al notebook
> [`Zoom_Resize_Image.ipynb`](./Zoom_Resize_Image.ipynb).
>
> El notebook cubre los fundamentos (píxel, PPI/DPI) y la interpolación clásica
> (Nearest, Bilineal, Bicúbica, Lanczos). Este documento profundiza en **por qué** esos
> métodos tienen un techo de calidad y en las técnicas modernas de **Super-Resolución de
> Imagen Única** (Single Image Super-Resolution, **SISR**): desde la idea de "aprender de
> ejemplos" (RAISR), pasando por las redes convolucionales y los GAN (Real-ESRGAN), hasta
> los **Transformers** (SwinIR / Swin2SR).

---

## Índice

1. [Recordatorio: el problema del reescalado](#1-recordatorio-el-problema-del-reescalado)
2. [Interpolación clásica: la matemática y su techo](#2-interpolación-clásica-la-matemática-y-su-techo)
3. [Formulación de la Super-Resolución](#3-formulación-de-la-super-resolución)
4. [El modelo de degradación (degradation pipeline)](#4-el-modelo-de-degradación-degradation-pipeline)
5. [Super-Resolución basada en ejemplos: del diccionario a RAISR](#5-super-resolución-basada-en-ejemplos-del-diccionario-a-raisr)
6. [Super-Resolución con redes convolucionales (CNN)](#6-super-resolución-con-redes-convolucionales-cnn)
7. [El salto perceptual: SRGAN, ESRGAN y Real-ESRGAN](#7-el-salto-perceptual-srgan-esrgan-y-real-esrgan)
8. [Transformers para imágenes: ViT, SwinIR y Swin2SR](#8-transformers-para-imágenes-vit-swinir-y-swin2sr)
9. [Métricas de calidad y el dilema percepción–distorsión](#9-métricas-de-calidad-y-el-dilema-percepción-distorsión)
10. [Más allá de la guía: difusión, escala arbitraria y otras familias](#10-más-allá-de-la-guía-difusión-escala-arbitraria-y-otras-familias)
11. [Notas prácticas (Colab / GPU T4)](#11-notas-prácticas-colab--gpu-t4)
12. [Referencias](#12-referencias)

---

## 1. Recordatorio: el problema del reescalado

Ampliar (*upscaling*) una imagen significa estimar los valores de píxeles que **no existían**
en la imagen original. Si una imagen de baja resolución (LR) de tamaño `W×H` se amplía un
factor `s`, hay que generar `s²·W·H − W·H` píxeles nuevos. El reescalado es, por tanto, un
**problema mal planteado** (*ill-posed*): para una misma imagen LR existen infinitas imágenes
de alta resolución (HR) que, al reducirse, producirían esa LR. Cualquier método de ampliación
no es más que una forma de **elegir una** de esas infinitas soluciones.

- La **interpolación clásica** elige asumiendo que la imagen es *suave* (los píxeles nuevos
  son combinaciones de los vecinos). Es una *prior* matemática fija.
- La **super-resolución aprendida** elige usando una *prior* aprendida de datos reales: "las
  imágenes naturales tienen bordes, texturas, patrones de piel/madera/pelo...". Por eso puede
  "alucinar" detalle plausible que la interpolación nunca generaría.

---

## 2. Interpolación clásica: la matemática y su techo

El notebook compara cuatro métodos. Todos son **convoluciones con un núcleo separable**
`k(x,y) = k(x)·k(y)`, así que basta con definir el núcleo 1D `k(x)` (la distancia `x` se mide
en unidades de píxel de la rejilla origen). Ampliar un factor `s` equivale a muestrear la señal
reconstruida `Σ_n I[n]·k(x − n)` en las nuevas posiciones fraccionarias.

### Nearest-Neighbor
Asigna a cada píxel destino el valor del píxel origen más cercano. Su núcleo es la **función
caja** (*box*):

$$ k_{\text{near}}(x) = \begin{cases} 1 & |x| < 0.5 \\ 0 & \text{en otro caso} \end{cases} $$

Soporte de 1 píxel, coste `O(1)` por muestra. En frecuencia su respuesta es un `sinc` muy
ancho: deja pasar mucho *aliasing*, de ahí los "escalones" y bloques. Preserva bordes duros
sin halos y **no inventa valores** (útil para máscaras, mapas de etiquetas o *pixel art*, donde
interpolar sería un error).

### Bilineal
Interpolación lineal en X y luego en Y ⇒ media ponderada de los **4** vecinos, con pesos
proporcionales al área opuesta. Núcleo **triangular** (*tent*), soporte de 2 píxeles:

$$ k_{\text{lin}}(x) = \max(0,\; 1 - |x|) $$

Es `C⁰` (continua pero con derivada discontinua). Atenúa las altas frecuencias de forma suave
→ elimina casi todo el *aliasing* pero **emborrona**. Coste bajo (4 muestras por píxel).

### Bicúbica
Núcleo cúbico por tramos de **Keys**, soporte de 4 píxeles (vecindario **4×4** = 16 muestras en
2D). Con el parámetro `a` (habitualmente `a = −0.5`, que hace el núcleo tangente al `sinc` en el
origen):

$$ k_{\text{cub}}(x) = \begin{cases}
(a+2)|x|^3 - (a+3)|x|^2 + 1 & |x| \le 1 \\
a|x|^3 - 5a|x|^2 + 8a|x| - 4a & 1 < |x| < 2 \\
0 & |x| \ge 2 \end{cases} $$

Es `C¹` (derivada continua) y su respuesta en frecuencia es más plana en la banda de paso y cae
más rápido que la bilineal → **mejor preservación de detalle** sin escalones. Con `a = −0.75`
(convención de OpenCV) enfatiza algo más los bordes. Es el estándar de facto en fotografía.

### Lanczos
Aproxima el reconstructor ideal (`sinc`) recortándolo con una **ventana** `sinc(x/a)` para
hacerlo de soporte finito (`a = 2` o `3`; en OpenCV `INTER_LANCZOS4` usa `a = 4`):

$$ k_{\text{lanczos}}(x) = \begin{cases}
\operatorname{sinc}(x)\,\operatorname{sinc}(x/a) & |x| < a \\
0 & |x| \ge a \end{cases}, \qquad \operatorname{sinc}(x) = \frac{\sin(\pi x)}{\pi x} $$

El `sinc` es el filtro paso-bajo ideal (recuperación perfecta de una señal de banda limitada,
teorema de Nyquist–Shannon), así que Lanczos da la **banda de paso más plana** y los **bordes
más nítidos** de los cuatro. El precio: sus lóbulos negativos generan *overshoot* → **halos**
(*ringing*) cerca de bordes de alto contraste, y su soporte grande lo hace el más costoso.

### Comparativa rápida

| Método | Soporte 1D | Vecinos 2D | Continuidad | Efecto dominante |
| :--- | :---: | :---: | :---: | :--- |
| Nearest | 1 px | 1 | discontinua | bloques / aliasing |
| Bilineal | 2 px | 4 | `C⁰` | emborronamiento |
| Bicúbica | 4 px | 16 | `C¹` | equilibrio nitidez/suavidad |
| Lanczos | 2a px | (2a)² | `C∞` a tramos | nitidez + posible ringing |

### El techo común

Todos son **filtros lineales de convolución con un núcleo fijo**. La salida es una
combinación lineal de píxeles de entrada:

$$ I_{HR}(x,y) = \sum_{i,j} k(i,j)\, I_{LR}(x_i, y_j) $$

Como `k` no depende del contenido de la imagen, **no puede añadir frecuencias nuevas**: solo
redistribuye la información existente. Matemáticamente, la energía en altas frecuencias que se
perdió al reducir la imagen **no se puede recuperar** con un filtro lineal. De ahí el
emborronamiento inevitable. Romper ese techo exige una *prior* no lineal y dependiente del
contenido: eso es lo que aporta el aprendizaje.

📎 Lectura recomendada sobre Lanczos: <https://mazzo.li/posts/lanczos.html>

---

## 3. Formulación de la Super-Resolución

La SISR busca una función `F` que, dada una imagen LR, estime la HR:

$$ \hat{I}_{HR} = F(I_{LR}; \theta) $$

donde `θ` son parámetros aprendidos minimizando una pérdida sobre un conjunto de pares
`(I_{LR}, I_{HR})`:

$$ \theta^* = \arg\min_{\theta} \; \frac{1}{N}\sum_{n=1}^{N} \mathcal{L}\big(F(I_{LR}^{(n)};\theta),\, I_{HR}^{(n)}\big) $$

La pieza clave es: **¿de dónde salen esos pares de entrenamiento?** Casi nunca se tienen
fotos reales del mismo objeto a dos resoluciones perfectamente alineadas. La solución
—exactamente la intuición de "downgradear imágenes buenas"— es el modelo de degradación.

---

## 4. El modelo de degradación (degradation pipeline)

Se parte de imágenes HR de alta calidad y se generan sus versiones LR aplicando una
degradación conocida. El modelo clásico ("bicubic degradation") es:

$$ I_{LR} = (I_{HR} * k)\downarrow_s $$

es decir, **desenfoque** (convolución con un núcleo `k`) seguido de **submuestreo** por un
factor `s`. El modelo realista (BSRGAN / Real-ESRGAN) lo amplía con una cadena más rica:

$$ I_{LR} = \big[\, (I_{HR} * k)\downarrow_s + n \,\big]_{\text{JPEG}} $$

añadiendo **ruido** `n` (gaussiano, de Poisson, del sensor) y **artefactos de compresión
JPEG**, a menudo aplicados varias veces ("*high-order degradation*") para imitar imágenes del
mundo real (capturas, fotos antiguas, vídeo recomprimido).

> En el notebook, la función `degrade()` implementa la versión simple (achicar + reampliar
> con bicúbica) para ilustrar el concepto y generar el par de entrenamiento de la "tabla".

### Anatomía de cada bloque

- **Desenfoque `k`.** No es un solo núcleo: se muestrea de una familia (gaussiano isotrópico y
  **anisotrópico**, más núcleos *generalized/plateau* Gaussian). El desenfoque anisotrópico
  imita el desenfoque de movimiento y las lentes reales. Modela el *point spread function* del
  sistema óptico.
- **Submuestreo `↓s`.** Se elige aleatoriamente entre *area*, bilineal y bicúbico, e incluso a
  veces se **sobre-muestrea** antes para variar la escala efectiva. Cada modo deja una firma de
  *aliasing* distinta que el modelo aprende a revertir.
- **Ruido `n`.** Mezcla de **gaussiano** (aditivo, del sensor en poca luz), de **Poisson**
  (dependiente de la señal, granulado fotónico) y ruido en escala de grises o por canal.
- **JPEG.** Compresión con *quality factor* aleatorio: introduce **bloques 8×8** y *ringing*
  alrededor de bordes por la cuantización DCT. Es el artefacto más común en imágenes de internet.

### High-order degradation (Real-ESRGAN)

El salto clave de Real-ESRGAN es aplicar la cadena **dos veces en serie** (*second-order*):

$$ I_{LR} = \mathcal{D}_2\big(\mathcal{D}_1(I_{HR})\big), \qquad
\mathcal{D}_i = \big[\,(\,\cdot\, * k_i)\downarrow_{s_i} + n_i\,\big]_{\text{JPEG}_i} $$

Una sola pasada no captura imágenes que ya venían degradadas (una foto de internet reescalada y
recomprimida varias veces). Encadenar dos degradaciones aleatorias amplía enormemente el
espacio de degradaciones "vistas". Además se añade un **filtro `sinc`** al final para modelar el
*ringing* y el *overshoot* típicos de reescalados previos.

**Por qué importa:** el modelo aprendido solo sabe deshacer las degradaciones que vio en
entrenamiento. Un modelo entrenado solo con "bicubic degradation" funciona mal con fotos
reales ruidosas; por eso Real-ESRGAN invierte tanto esfuerzo en una degradación realista.
El riesgo opuesto: una degradación **demasiado** agresiva hace que el modelo alucine y sobre-
suavice imágenes que en realidad estaban limpias.

📎 BSRGAN (degradación aleatoria): <https://arxiv.org/abs/2103.14006> · Real-ESRGAN (high-order): <https://arxiv.org/abs/2107.10833>

---

## 5. Super-Resolución basada en ejemplos: del diccionario a RAISR

Es la familia a la que pertenece, conceptualmente, la "tabla de vecindarios" del notebook.

### Example-based SR (Freeman et al., 2002)
Se construye una base de datos de **parches** (*patches*) emparejados LR↔HR. Para ampliar,
por cada parche LR de la imagen de entrada se busca el parche LR más parecido del diccionario
y se pega su contraparte HR. Funciona, pero:

- La base de datos es enorme y la búsqueda del vecino más cercano es lenta.
- **La maldición de la dimensionalidad**: el número de configuraciones posibles de un
  vecindario crece exponencialmente con su tamaño. Con 8 vecinos en escala de grises ya hay
  `256⁸ ≈ 1.8·10¹⁹` combinaciones; en color es astronómico. La tabla nunca se puede llenar
  ni almacenar → se queda vacía justo donde la consultas.

### Sparse coding / dictionary learning (Yang et al., 2010)
En vez de guardar todos los parches, aprende un **diccionario compacto** y representa cada
parche como combinación *sparse* de sus átomos. Comprime el problema, pero sigue siendo lento.

### RAISR (Google, 2016) — la idea "apañada"
**Rapid and Accurate Image Super-Resolution.** En lugar de una tabla con todos los colores,
clasifica (*hashing*) cada parche en un número pequeño de **cubos (*buckets*)** según
características de su **gradiente local**, y aprende un **filtro lineal óptimo por cubo**.

**El hash por gradiente.** Para el parche se calcula el **tensor de estructura** (la matriz de
covarianza de los gradientes `∇I` ponderada por una gaussiana). Sus dos autovalores `λ₁ ≥ λ₂` y
el autovector principal dan tres descriptores, cada uno cuantizado en pocos niveles:

- **Ángulo** `θ` (orientación del borde) → p. ej. 24 cubos.
- **Fuerza** (*strength*) `≈ √λ₁` (magnitud del gradiente) → p. ej. 3 cubos.
- **Coherencia** `(√λ₁ − √λ₂)/(√λ₁ + √λ₂) ∈ [0,1]` (¿es un borde limpio o textura?) → 3 cubos.

Eso da del orden de `24 × 3 × 3 = 216` cubos (× el factor de escala), **muchísimos menos** que
las `256⁸` de la tabla ingenua, pero **informados por la geometría del borde** en vez de por el
color crudo.

**Aprender el filtro de cada cubo.** Por cada cubo se resuelve un problema de **mínimos
cuadrados**: se busca el filtro `h` que mejor mapea los parches LR de ese cubo a su píxel HR
central,

$$ h^\* = \arg\min_h \sum_{p \in \text{cubo}} \big( h^\top p_{LR} - p_{HR} \big)^2
       = (A^\top A)^{-1} A^\top b $$

donde las filas de `A` son los parches LR y `b` los píxeles HR objetivo. Es la **solución
óptima cerrada** (ecuación normal), no un descenso de gradiente.

**Inferencia** (muy barata): para cada píxel → gradiente → índice de cubo → producto escalar con
su filtro. Google añade además *blending* con **census/coherencia** para suavizar transiciones y
evitar artefactos. Resultado: calidad cercana a las CNN de su época, **~100× más rápido** que
ellas y con un modelo de unos pocos KB.

> RAISR es prácticamente la versión funcional de la "tabla de vecindarios con cajas". Dos ideas
> lo hacen viable frente a la tabla ingenua: (1) **cuantizar por geometría del gradiente** en
> vez de por color, y (2) **aprender un filtro lineal por cubo** en lugar de memorizar un color.
> La mini-implementación del notebook (cuantizar el vecindario 3×3 en `Q` niveles y aprender el
> color medio por caja) es un RAISR de juguete: usa (1) de forma tosca y sustituye (2) por la
> media, para mostrar por qué la agrupación en pocas cajas es lo que rompe el muro de la tabla.

📎 RAISR: <https://arxiv.org/abs/1606.01299>

---

## 6. Super-Resolución con redes convolucionales (CNN)

La idea de "aprender la *prior* de las imágenes naturales" cristaliza con el deep learning.
Una CNN **no guarda una tabla**: guarda una función comprimida en sus pesos que **generaliza**
a vecindarios nunca vistos. Hitos:

| Modelo | Año | Aportación principal |
| :--- | :--- | :--- |
| **SRCNN** | 2014 | Primera CNN para SR. 3 capas: extracción de parches, mapeo no lineal, reconstrucción. Demostró que una red supera a la interpolación. |
| **FSRCNN** | 2016 | Procesa en resolución LR y amplía al final → mucho más rápido. |
| **ESPCN** | 2016 | Introduce el ***sub-pixel convolution* / *pixel shuffle***: la red aprende `s²` mapas y los reordena para ampliar, sin interpolar antes. Estándar hoy. |
| **VDSR** | 2016 | Red muy profunda (20 capas) + **aprendizaje residual** (predice solo el detalle que falta). |
| **EDSR** | 2017 | Bloques residuales sin BatchNorm; ganador del reto NTIRE. Gran calidad en PSNR. |
| **RCAN** | 2018 | **Atención de canal** + redes muy profundas (400+ capas). |

### Los mecanismos clave en detalle

**SRCNN (3 capas).** Formaliza la SR como tres operaciones aprendidas de golpe: `9×9` conv
(*extracción de parches* → mapa de features), `1×1` conv (*mapeo no lineal* LR→HR en el espacio
de features) y `5×5` conv (*reconstrucción*). Trabaja sobre la imagen **ya interpolada** con
bicúbica (por eso es lento: opera a resolución HR).

**FSRCNN — procesar en LR.** Estructura en "reloj de arena": *shrinking* `1×1` para reducir
canales, varias conv pequeñas de mapeo, *expanding*, y al final una **deconvolución** que hace
el *upscaling*. Al trabajar casi todo en resolución LR es **decenas de veces más rápido** que
SRCNN.

**ESPCN — sub-pixel convolution / pixel shuffle.** En vez de interpolar antes, la red produce
`s²` mapas de features a resolución LR y los **reordena** (*PixelShuffle*) en un solo mapa `s`
veces mayor:

$$ \text{PS}: \; (C\cdot s^2)\times H\times W \;\longrightarrow\; C\times (sH)\times (sW) $$

Cada bloque de `s²` canales se "despliega" en una vecindad `s×s`. Ventajas frente a la
deconvolución: **sin artefactos de tablero** (*checkerboard*) y cómputo en LR. Es el módulo de
*upscaling* estándar hoy (lo usan SRGAN, EDSR, etc.).

**VDSR — aprendizaje residual.** Con 20 capas, en vez de predecir la HR completa predice solo el
**residuo** (el detalle que falta): `Î_HR = I_bicúbica + R(I_bicúbica)`. La red modela una señal
casi nula salvo en los bordes → converge mucho mejor. Requirió *gradient clipping* para permitir
tasas de aprendizaje altas en una red tan profunda.

**EDSR — limpiar el bloque residual.** Quita la **BatchNorm** de los bloques residuales (en SR,
BN normaliza y "aplana" el rango de features, perjudicial cuando la salida debe ser fiel al color
de entrada). Eso libera memoria para hacer la red más ancha/profunda, con **residual scaling**
(multiplicar la rama residual por ~0.1) para estabilizar el entrenamiento.

**RCAN — atención de canal (CA) + RIR.** Introduce *Residual-in-Residual* (residuos anidados con
*long skip connections*) para poder apilar 400+ capas, y **atención de canal**: por cada canal se
calcula un descriptor global (*global average pooling*) y una pequeña red *sigmoid* produce un
peso `∈[0,1]` que **recalibra** ese canal:

$$ s = \sigma\big(W_2\,\delta(W_1\, z)\big), \qquad z_c = \tfrac{1}{HW}\textstyle\sum_{i,j} x_c(i,j),
   \qquad \hat{x}_c = s_c\, x_c $$

La red aprende a **dar más peso a los canales que codifican alta frecuencia** y a atenuar los de
baja información. (RCAB = bloque residual + esta rama de atención.)

**Funciones de pérdida.** Las primeras CNN minimizan el error píxel a píxel:

$$ \mathcal{L}_{1} = \lVert I_{HR} - \hat{I}_{HR} \rVert_1 \qquad \mathcal{L}_{2} = \lVert I_{HR} - \hat{I}_{HR} \rVert_2^2 $$

Maximizan PSNR pero tienden a dar imágenes **suaves** (la media de todas las HR plausibles es
borrosa). Resolver eso lleva al siguiente salto.

---

## 7. El salto perceptual: SRGAN, ESRGAN y Real-ESRGAN

### Por qué las pérdidas L1/L2 emborronan
Para una LR dada hay muchas HR plausibles (§1). Minimizar el error medio empuja a `F` hacia la
**media** de todas ellas, y la media de texturas alineadas de forma distinta es **borrosa**. Un
detalle nítido pero ligeramente desplazado penaliza *más* en L2 que un promedio suave, así que el
óptimo por MSE es literalmente "no arriesgar". Romperlo exige una señal que premie el **realismo**
aunque el detalle no coincida píxel a píxel. Eso son la pérdida perceptual y la adversaria.

### SRGAN (2017)
Introduce dos ideas para imágenes **perceptualmente** nítidas en vez de óptimas en PSNR:

- **Pérdida perceptual** (*perceptual / content loss*): en vez de comparar píxeles, compara
  *features* de una red **VGG-19** preentrenada, `L_perc = ‖φ(I_HR) − φ(Î_HR)‖²`, donde `φ` es
  la activación de una capa intermedia. Premia que las **texturas y estructuras** "parezcan"
  correctas, no que coincida cada píxel.
- **Pérdida adversaria** (GAN): un **discriminador** `D` aprende a distinguir HR reales de
  generadas, y el generador `G` aprende a engañarlo. El discriminador es, en esencia, el
  "juez que detecta inventos que no parecen reales" → es el equivalente aprendido del sistema
  de "filtros vigilantes por consenso".

$$ \mathcal{L}_{G} = \underbrace{\mathcal{L}_{perceptual}}_{\text{textura realista}}
   + \lambda\,\underbrace{\mathcal{L}_{adv}}_{\text{engañar a }D}
   + \eta\,\underbrace{\mathcal{L}_{1}}_{\text{fidelidad de color}} $$

**Arquitectura.** El generador es una red residual (bloques *conv-BN-PReLU*) que trabaja en LR y
amplía al final con **sub-pixel conv** (§6). El discriminador es una CNN tipo VGG que termina en
una probabilidad "real/falsa". El término L1/perceptual evita que el color derive mientras la
parte adversaria añade el detalle de alta frecuencia.

### ESRGAN (2018)
Mejora SRGAN en tres frentes:

- **Bloque RRDB** (*Residual-in-Residual Dense Block*): combina conexiones **densas** (cada conv
  ve la salida de todas las anteriores) con residuos anidados y **sin BatchNorm** (que en SR
  introduce artefactos). Usa **residual scaling** `β ≈ 0.2` en cada rama para estabilizar redes
  muy profundas.
- **Discriminador relativista** (RaGAN): en lugar de estimar "¿es real?" (absoluto), estima
  **"¿es esto más realista que aquello?"** (relativo): `D(x_r, x_f) = σ(C(x_r) − 𝔼[C(x_f)])`.
  Da gradientes más informativos y texturas más agudas.
- **Pérdida perceptual antes de la activación** (*pre-activation VGG features*): las features
  pre-ReLU conservan más información de brillo y bordes → texturas más realistas.

### Real-ESRGAN (2021) — el que usa el notebook
Adapta ESRGAN al **mundo real** (mismo generador RRDB) combinando:

1. **Degradación de alto orden** (§4): desenfoque + ruido + JPEG + `sinc` aplicados **dos veces
   en serie**, para que el modelo sepa restaurar fotos reales degradadas, no solo "bicubic".
2. Un **discriminador U-Net con spectral normalization**. A diferencia del discriminador global
   de ESRGAN, el U-Net emite un **mapa de realismo por píxel** (feedback local que distingue qué
   zonas parecen falsas), y la *spectral norm* (limitar la norma espectral de los pesos) **estabiliza**
   el entrenamiento y frena artefactos cuando la degradación es agresiva.

Es el estándar práctico para *upscaling* de fotos y arte (existe **Real-ESRGAN-anime**, con un
generador más ligero, para ilustración). Pesos x4 listos para usar; soporta *tiling* para
imágenes grandes.

📎 ESRGAN: <https://arxiv.org/abs/1809.00219> · Real-ESRGAN: <https://arxiv.org/abs/2107.10833> · Repo: <https://github.com/xinntao/Real-ESRGAN>

---

## 8. Transformers para imágenes: ViT, SwinIR y Swin2SR

### De ViT al problema de la super-resolución
El **Vision Transformer (ViT, 2020)** trocea la imagen en *parches*, los proyecta a *tokens* y
aplica **auto-atención**. Cada token genera *query*, *key* y *value*, y su salida es una media de
los valores ponderada por la similitud query·key:

$$ \text{Attn}(Q,K,V) = \operatorname{softmax}\!\Big(\tfrac{QK^\top}{\sqrt{d}} + B\Big)V $$

La ventaja frente a la convolución es el **contexto global**: cada token puede "mirar" a todos
los demás → modela dependencias de **largo alcance** (relaciona zonas lejanas), mientras que una
conv solo ve su vecindario local. Conecta con la intuición de "mirar más contexto para decidir
mejor un píxel", pero aprendida y sin tabla.

El problema: con `n` tokens, `QKᵀ` es `n×n` ⇒ coste y memoria **`O(n²)`**. A resolución de imagen
completa (miles de tokens) es inviable.

### Swin Transformer — atención por ventanas
Resuelve el coste partiendo la imagen en **ventanas** de `M×M` tokens (p. ej. `M = 8`) y
calculando atención **solo dentro** de cada ventana (**W-MSA**). El coste pasa de cuadrático en
todos los píxeles a **lineal**:

$$ \Omega(\text{MSA}) = O\big((HW)^2\big) \;\longrightarrow\; \Omega(\text{W-MSA}) = O\big(M^2\cdot HW\big) $$

Pero ventanas fijas nunca dejan que la información cruce sus bordes. La solución son las
**ventanas desplazadas** (*shifted windows*, **SW-MSA**): en capas alternas la rejilla de
ventanas se desplaza `M/2`, de modo que tokens que antes estaban en ventanas distintas ahora
comparten ventana. Se implementa con un **desplazamiento cíclico** + una **máscara de atención**
(para no mezclar regiones no adyacentes) que mantiene el coste. Alternar bloques W-MSA y SW-MSA
propaga el contexto por toda la imagen en pocas capas. Añade además un **sesgo de posición
relativa** `B` (el término del softmax de arriba), que codifica dónde está cada token respecto a
los otros dentro de la ventana.

### SwinIR (2021)
Aplica Swin a la restauración de imagen (SR, *denoising*, *JPEG deartifacting*) con tres etapas:

1. **Extracción superficial**: una conv obtiene features iniciales.
2. **Extracción profunda**: una pila de **RSTB** (*Residual Swin Transformer Blocks*). Cada RSTB
   es una secuencia de **STL** (*Swin Transformer Layers*: W-MSA/SW-MSA + MLP con LayerNorm y
   conexiones residuales) cerrada por una conv y una **skip connection** larga. Mezclar atención
   (global) con conv (localidad/inductive bias) da lo mejor de ambos mundos.
3. **Reconstrucción**: fusiona features superficiales + profundas y amplía con **sub-pixel conv**.

Alcanza el estado del arte con **menos parámetros** que las CNN equivalentes (EDSR/RCAN), porque
la atención captura dependencias que a la conv le costarían muchas capas.

### Swin2SR (2022)
Actualiza SwinIR a **Swin Transformer V2**, que corrige inestabilidades al escalar:

- **Post-normalization + scaled cosine attention**: normaliza después del bloque y usa similitud
  **coseno** (en vez del producto escalar) para evitar que unos pocos tokens dominen el softmax
  → entrenamiento estable en redes/resoluciones grandes.
- **Log-spaced continuous position bias**: aprende el sesgo de posición con una pequeña MLP sobre
  coordenadas en escala logarítmica, lo que permite **transferir** a ventanas de distinto tamaño
  que en entrenamiento.
- Modo **compressed-input** pensado para entradas con **JPEG** (deartifacting + SR a la vez).

Está integrado en 🤗 **`transformers`** (`Swin2SRForImageSuperResolution` + `Swin2SRImageProcessor`),
por lo que se usa casi igual que un modelo de NLP — justo lo que hace el notebook.

**Coste / memoria.** La atención, aun por ventanas, consume bastante VRAM con imágenes
grandes. Recomendación: procesar **recortes** o trocear (*tiling*). En el notebook se aplica
sobre un recorte pequeño por eso.

📎 ViT: <https://arxiv.org/abs/2010.11929> · Swin: <https://arxiv.org/abs/2103.14030> · SwinIR: <https://arxiv.org/abs/2108.10257> · Swin2SR: <https://arxiv.org/abs/2209.11345> · Modelos: <https://huggingface.co/caidas>

---

## 9. Métricas de calidad y el dilema percepción–distorsión

Comparar métodos de SR no es trivial; conviene reportar varias métricas.

- **PSNR** (Peak Signal-to-Noise Ratio): derivado del MSE. Más alto = más fiel píxel a píxel.
  No correlaciona bien con la percepción humana (un resultado borroso puede tener alto PSNR).

$$ \text{PSNR} = 10\,\log_{10}\!\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right) $$

- **SSIM** (Structural Similarity): compara **luminancia**, **contraste** y **estructura** en
  ventanas locales, combinando medias `μ`, varianzas `σ²` y covarianza `σ_{xy}`:

$$ \text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}
   {(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)} $$

  Rango `[0, 1]` (1 = idénticas), más cercano a la percepción que el PSNR porque penaliza cambios
  **estructurales**, no solo energéticos. Variante multi-escala: **MS-SSIM**.
- **LPIPS** (Learned Perceptual Image Patch Similarity): pasa ambas imágenes por una red profunda
  (AlexNet/VGG), y mide la **distancia L2 entre sus activaciones** capa a capa, con pesos
  aprendidos para alinearse con juicios humanos: `d = Σ_l ‖w_l ⊙ (φ_l(x) − φ_l(y))‖²`. Captura
  diferencias de textura y semántica que PSNR/SSIM ignoran. **Más bajo = más parecido
  perceptualmente.** Hoy es la métrica perceptual de referencia.
- **NIQE / MANIQA / NRQM**: métricas **sin referencia** (*no-reference*), no necesitan la HR
  original; puntúan la "naturalidad" de la imagen a partir de estadísticas o de una red. Útiles
  para SR del mundo real donde **no hay *ground truth*** (justo el caso de Real-ESRGAN/Swin2SR
  sobre fotos reales).

**El dilema percepción–distorsión** (Blau & Michaeli, 2018): existe un compromiso
**fundamental** entre fidelidad (PSNR/SSIM altos) y realismo perceptual (LPIPS bajo). No se
pueden maximizar ambos a la vez. Por eso:

- Los modelos orientados a **PSNR** (EDSR, SwinIR-PSNR) dan resultados fieles pero más suaves.
- Los modelos **GAN/perceptuales** (ESRGAN, Real-ESRGAN) dan texturas nítidas y realistas,
  pero pueden "inventar" detalle que no estaba (peor PSNR). La elección depende del uso:
  fidelidad métrica (medicina, forense) vs. estética (fotografía, restauración).

📎 The Perception-Distortion Tradeoff: <https://arxiv.org/abs/1711.06077> · LPIPS: <https://arxiv.org/abs/1801.03924>

---

## 10. Más allá de la guía: difusión, escala arbitraria y otras familias

La progresión **interpolación → RAISR → CNN → GAN → Transformer** es la columna vertebral de la
SISR, pero no agota el campo. Estas familias no se cubren en el notebook y algunas son **estado
del arte hoy**. Vale la pena conocerlas para elegir la herramienta correcta.

### Modelos de difusión (SR generativa) — el paradigma dominante actual
En lugar de una sola pasada, parten de **ruido** y lo *denoisan* iterativamente **condicionando
en la LR**, muestreando de la distribución de HR plausibles en vez de promediarla (por eso
esquivan el emborronamiento del MSE, §7). Formalmente aprenden a invertir un proceso de difusión:

$$ x_{t-1} = \tfrac{1}{\sqrt{\alpha_t}}\Big(x_t - \tfrac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,
   \epsilon_\theta(x_t, t, I_{LR})\Big) + \sigma_t z $$

- **SR3** (2021) — pionero, *iterative refinement* sobre píxeles.
- **SRDiff**, **Stable Diffusion x4 upscaler**, **LDM** — difusión en **espacio latente** (más barata).
- **StableSR / DiffBIR / PASD / SeeSR** (2023) — reutilizan un **prior de difusión preentrenado**
  (SD) para restaurar fotos reales degradadas.
- **SUPIR** (2024) — a gran escala y **guiado por texto** sobre SDXL; realismo altísimo, pero
  puede alucinar contenido no presente en la LR.
- **ResShift / SinSR** (2023–24) — difusión **eficiente** (pocos o un solo paso), atacando su
  mayor defecto: la lentitud (decenas de pasos de muestreo).

> **Trade-off vs. GAN**: mejor realismo y menos artefactos, pero mucho **más lentos** en
> inferencia y con más tendencia a "inventar". La investigación reciente va sobre **acelerarlos**
> (destilación a 1–4 pasos).

### Super-Resolución de escala arbitraria (representaciones implícitas)
- **LIIF** (2021), **LTE**, **CiaoSR** — aprenden una **función continua** de la imagen
  (*implicit neural representation*): dado un punto `(x,y)` continuo devuelven su color, así que
  amplían a **cualquier factor** (×2.7, ×5.3…) con un único modelo, no solo a los `s` entrenados.

### Transformers más allá de Swin
- **HAT** (2023, *Hybrid Attention Transformer*) — combina atención de canal + de ventana +
  *overlapping cross-attention*; **líder actual en PSNR**.
- **IPT** (transformer preentrenado multi-tarea), **SRFormer**, **DAT**, **GRL**, **ELAN**.

### State-space models / Mamba
- **MambaIR** (2024) — usa *state-space models* de **complejidad lineal** como alternativa a la
  atención, buscando el contexto global del transformer sin su coste cuadrático.

### Zero-shot / aprendizaje interno (sin dataset externo)
- **ZSSR** (2018) y **Deep Image Prior** — se entrenan con la **propia imagen de test**,
  explotando la recurrencia de parches a distintas escalas. Útiles cuando no hay datos de
  entrenamiento o la degradación es desconocida.
- **KernelGAN** — estima el **kernel de degradación real** de la imagen (*blind SR*), que luego
  alimenta a un modelo de SR.

### Restauración de caras (nicho muy práctico)
- **GFP-GAN**, **CodeFormer** (codebook VQ + transformer), **GPEN**, **PULSE** — usan un **prior
  facial** (típicamente StyleGAN) para reconstruir rostros con identidad plausible. Es la
  tecnología detrás de las funciones de "realzar caras" de muchas apps.

### Fuera del alcance SISR (contexto)
- **Flow-based**: **SRFlow** (2020) — flujos normalizadores; modelan la distribución completa de
  HR y permiten muestrear salidas **diversas**.
- **Reference-based (RefSR)**: **TTSR**, **MASA-SR**, **C2-Matching** — usan una **segunda imagen
  HR** de apoyo (misma escena/objeto) para transferir textura real.
- **Video SR**: **EDVR**, **BasicVSR / BasicVSR++**, **RVRT** — explotan la **coherencia
  temporal** entre fotogramas (alineación/propagación) además de la información espacial.

📎 SR3: <https://arxiv.org/abs/2104.07636> · StableSR: <https://arxiv.org/abs/2305.07015> · DiffBIR: <https://arxiv.org/abs/2308.15070> · ResShift: <https://arxiv.org/abs/2307.12348> · SUPIR: <https://arxiv.org/abs/2401.13627> · LIIF: <https://arxiv.org/abs/2012.09161> · HAT: <https://arxiv.org/abs/2205.04437> · MambaIR: <https://arxiv.org/abs/2402.15648> · ZSSR: <https://arxiv.org/abs/1712.06087> · CodeFormer: <https://arxiv.org/abs/2206.11253> · SRFlow: <https://arxiv.org/abs/2006.14200> · BasicVSR++: <https://arxiv.org/abs/2104.13371>

---

## 11. Notas prácticas (Colab / GPU T4)

- **¿T4 suficiente?** Sí. Sus 16 GB de VRAM bastan para Real-ESRGAN x4 y Swin2SR sobre
  imágenes/recortes razonables. Activa la GPU en Colab: *Entorno de ejecución → Cambiar tipo
  de entorno → T4 GPU*.
- **Real-ESRGAN** y memoria: para imágenes grandes usa **tiling** (procesa la imagen por
  baldosas con solape). El repo expone un parámetro `tile` para ello.
- **Swin2SR** es el más sensible a la VRAM (atención ∝ píxeles). Mantén la entrada por debajo
  de ~512×512 px en una T4, o trocea. En el notebook se usa un recorte de 96 px → x4.
- **Espacio de color.** Mucha SR clásica opera sobre el canal de **luminancia (Y de YCbCr)**,
  donde vive el detalle que percibe el ojo, y amplía la crominancia por separado. Es la
  versión "buena" de la intuición de "separar luces/sombras del color" del planteamiento
  original.
- **Factor de escala.** Los modelos están entrenados para un `s` concreto (x2, x4...).
  Usar el modelo correcto importa; encadenar x2 dos veces no equivale a un x4 entrenado.
- **Formato de entrada.** Da al modelo la imagen **LR real**; no le pases una ya ampliada con
  bicúbica (salvo modelos que lo esperan), o degradarás el resultado.

---

## 12. Referencias

**Interpolación clásica**
- Lanczos resampling — <https://mazzo.li/posts/lanczos.html>
- Keys, *Cubic convolution interpolation* (1981).

**SR basada en ejemplos**
- Freeman, Jones, Pasztor, *Example-Based Super-Resolution* (2002).
- Yang et al., *Image Super-Resolution via Sparse Representation* (2010).
- Romano, Isidoro, Milanfar, **RAISR** (2016) — <https://arxiv.org/abs/1606.01299>

**SR con CNN**
- Dong et al., **SRCNN** (2014) — <https://arxiv.org/abs/1501.00092>
- Shi et al., **ESPCN** / sub-pixel conv (2016) — <https://arxiv.org/abs/1609.05158>
- Kim et al., **VDSR** (2016) — <https://arxiv.org/abs/1511.04587>
- Lim et al., **EDSR** (2017) — <https://arxiv.org/abs/1707.02921>
- Zhang et al., **RCAN** (2018) — <https://arxiv.org/abs/1807.02758>

**SR perceptual / GAN**
- Ledig et al., **SRGAN** (2017) — <https://arxiv.org/abs/1609.04802>
- Wang et al., **ESRGAN** (2018) — <https://arxiv.org/abs/1809.00219>
- Wang et al., **Real-ESRGAN** (2021) — <https://arxiv.org/abs/2107.10833> · <https://github.com/xinntao/Real-ESRGAN>
- Zhang et al., **BSRGAN** (2021) — <https://arxiv.org/abs/2103.14006>

**Transformers**
- Dosovitskiy et al., **ViT** (2020) — <https://arxiv.org/abs/2010.11929>
- Liu et al., **Swin Transformer** (2021) — <https://arxiv.org/abs/2103.14030>
- Liang et al., **SwinIR** (2021) — <https://arxiv.org/abs/2108.10257>
- Conde et al., **Swin2SR** (2022) — <https://arxiv.org/abs/2209.11345> · <https://github.com/mv-lab/swin2sr>

**Métricas**
- Blau, Michaeli, *The Perception-Distortion Tradeoff* (2018) — <https://arxiv.org/abs/1711.06077>
- Zhang et al., **LPIPS** (2018) — <https://arxiv.org/abs/1801.03924>

**Herramientas / modelos preentrenados**
- 🤗 Transformers (Swin2SR) — <https://huggingface.co/docs/transformers/model_doc/swin2sr>
- Modelos `caidas/swin2SR-*` — <https://huggingface.co/caidas>
- Real-ESRGAN (wrapper sencillo) — <https://github.com/ai-forever/Real-ESRGAN>
