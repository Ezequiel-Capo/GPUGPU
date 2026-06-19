# Practico 4 (GPU)

## Ejecutar todos los casos de prueba

```bash
bash prueba.sh
```

## Ejecución puntual

### Ejecución con nsys, ver tiempos de ejecuciones
```bash

sbatch --parsable --output=salida.out --error=salida.err lanzar.sh "./ej" --nsys thrust x y ...
```

### Ejecución sin nsys
```bash

sbatch lanzar.sh "./ej" x y ...

```

- Donde x,y,... parametros de cada código 
- "./ej" es alguno de los vistos en lanzar.sh

**Aclaración:**
- Se formatea la salida del reporte nsys, para que solo muestre las métricas de ejecución pertinentes
- El tiempo de ejecucion de prueba.sh es aproximadamente de 5min, se brindan salidas ya dispuestas en bash salidas.sh, si solo se qieren reiterar resultados

## Formatear .sh
- sed -i 's/\r$//' *