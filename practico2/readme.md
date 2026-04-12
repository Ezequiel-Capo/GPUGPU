# Practico 2 (GPU)

Este directorio incluye:

- `ej1.cu`
- `ej2.cu`
- `ej3.cu`
- `lanzar.sh` (compila y ejecuta en SLURM)
- `prueba.sh` (envia casos de prueba)
- En caso de falla por fin de línea emplear:

```
 sed -i 's/\r$//' *
```

**Aclaración:**
Existen ocasiones que prueba.sh no encuentra los archivos de salida, especialmente ej1.out, suponemos sucede por la demora en escribir dicho archivo. Revisarlo manual, si sucede:
```
 cat ej1.out
```
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
sbatch lanzar.sh ./ej2 p1 p2 p3 p4 p5 p6 p7 p8
```

- `p1`: tamaño matriz
- `p2`: i1
- `p3`: j1
- `p4`: i2
- `p5`: j2
- `p6`: valor de adición
- `p7`: tamano de bloque en X (`blockDim.x`) (solo 1D)
- `p8`: tamano de bloque en Y (`blockDim.y`)

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

**Ejemplo:**

```bash
sbatch lanzar.sh ./ej3 1024 1024 32 32
```
