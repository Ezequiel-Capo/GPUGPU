# Laboratorio GPGPU

## Implementaciones Multiplicacion XXT 

- WMMA uint4,uint4,int
- WMMA uint8,uint8,int
- __dp4a empaquetado uint2
- cuBlas Syrk

## Lanzar jobs de pruebas
``` bash prueba_performance.sh``` 

## Visualizar salidas de jobs de pruebas performance
``` bash salidas.sh``` 

## Comparación error numérico vs Cublas
``` bash error_numerico.sh```

## Lanzar jobs de parametria WMMA U4 y U8
``` bash prueba_parametria.sh```

### Salidas preparadas

Los casos de prueba disponen tiempos de ejecución extensos:
- Se reiteran a varias implementaciones
- Varios tamaños de matriz (relativamente grandes) 
- 11 Ejecuciones (1 warm-up, 10 para tomar el promedio)

Es por ello se prepararon archivos para visualizar la salida ya preparada:
- salidas_parametria.txt 
- salidas.txt

Duración de todas las ejecuciones (ErrorNumerico, Performance, Parametria) alrededor de 16minutos. 

