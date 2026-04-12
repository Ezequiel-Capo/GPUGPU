
# Practico 2 (GPU)

Este directorio incluye:
- `ej1.cu`
- `ej2.cu`
- `ej3.cu`
- `lanzar.sh` (compila y ejecuta en SLURM)
- `prueba.sh` (envia casos de prueba)

## Ejecutar todos los casos de prueba

```bash
bash prueba.sh
```

El script envia jobs para `ej1`, `ej2`, `ej3`, espera su finalizacion y luego muestra los resultados de salida.

## Ejecutar un ejercicio individual

`lanzar.sh` compila y ejecuta el programa indicado. Actualmente soporta:
- `./ej1`
- `./ej2`
- `./ej3`

### Ejercicio 1

```bash
sbatch lanzar.sh ./ej1 secreto.txt
```

### Ejercicio 2

```bash
sbatch lanzar.sh ./ej2 xxx
```

### Ejercicio 3

```bash
sbatch lanzar.sh ./ej3 p1 p2 p3 p4 [v]
```

Parametros:
- `p1`: dimension X (ancho)
- `p2`: dimension Y (alto)
- `p3`: tamano de bloque en X (`blockDim.x`)
- `p4`: tamano de bloque en Y (`blockDim.y`)
- `v`: opcional, modo verbose (imprime matriz original y traspuesta; no recomendado para tamanos grandes)

Ejemplo:

```bash
sbatch lanzar.sh ./ej3 1024 1024 32 32
```


