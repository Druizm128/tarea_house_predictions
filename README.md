# Tara 02 de predicción de precios de casas

## Working backwards

### 5 preguntas del cliente

* ¿Qué quiere el cliente?

Comparar precios de compra ventas de casas para conocer si los precios de 
mercado están sobre o subvaluados, utilizando las características básicas
de las propiedades.

* ¿Cuál es el problema u oportunidad?

Cuando estas buscando una propiedad para comprar o vender, es dificil saber si
un precio es razonable. Para ello se hacen avalúos de las propiedades, sin 
embargo el costo de transacción de descubrir un precio de mercado no lo pueden
pagar todos o no lo puedes pagar para cada propiedad de interés.

Poder utilizar datos del mercado sobre propiedades en venta, utilizando 
características fundamentales (superficie, estacionamiento, cuartos, baños, etc).
Podría ayudar a estimar un precio promedio para cada inmueble.

* ¿Por qué esto es importante para el cliente?

El usuario de la aplicación, busca poder tomar mejores decisiones, para lo
cual requiere insumos, para poder refinar su proceso de decisión. 

Poder brindar al cliente un precio promedio utilizando su celular e ingresando
a lo mucho 5 características de la propiedad sería muy beneficioso.

* ¿Cuál es el beneficio del cliente?

Contar con una estimación del precio de una propiedad (una estimación second
best al de un avalúo) en segundos. Esto para cada propiedad en la que este
interesado.

* ¿Cómo se ve la experiencia del cliente?

Una aplicación en el teléfono o internet, en la que como una calculadora se 
ingrese la información y se genere una estimación.

### FAQ

* ¿Esta aplicación se puede usar en cualquier ciudad?

No, está es una prueba de concepto para demostrar la funcionalidad de la 
aplicación. Estamos utilizando un dataset que se puede aproximar a alguno que
se pueda levantar en cualquier ciudad del mundo. En este caso se utilizan
datos de la ciudad de AMES, IOWA.

* ¿Qué mecanismo se utilizó para estimar los precios?

Machine learning, en el escenario base una regresión lineal del precio de venta
vs superificie, # de habitacioones, # de baños, # de medios baños, # de capacidad
de autos en el garage.

* ¿Se puede estimar el precio con otros mecanismos?

Sí, como regresión bayesiana, bootstrap regression, promedios, etc. En esta 
POC utilizamos un modelo sencillo, no estamos enfocados en la precisión de la
estimación, sino en demostrar la usabilidad de la herramienta.

* ¿El precio estimado es una métrica razonable?

Sí, en el sentido que la regresión está calculando un precio promedio, según
los precios y características que se vieron en entrenamiento. Sin embargo,
sería conveniente en una segunda iteración tener una estimación de los intervalos
de confianza de esa estimación.

    - Intervalos bayesianos del estimador de la media.
    - Bootstrap para calcular los intervalos.
    - Conformal prediction.

Se deja este feature para una siguiente iteración.

* ¿Cómo puedo utilizar la herramienta?

La herramienta de estimación está construida en un StreamLit que aún no está
publicado en internet. En la práctica el backend de la herramienta sería en
Python, y el front end, puede construirse con cualquier otro lenguaje y framework
que permita construir una aplicación más responsiva y veloz.

* ¿Qué puede hacer la herramienta?

1) Estimar precios promedio por colonia (Hardcoded inference).
2) Estimar el precio de una sola casa, a partir de ingresar sus características (Real Time Inference)
3) Estimar el precio de un conjunto de casas a partir de la ingestión de un CSV. (Batch Inference)