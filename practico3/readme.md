# Practico 3 (GPU)

Este directorio incluye:

- `ej1.cu`
- `ej2.cu`
- `ej2_2.cu`
- `lanzar.sh` (compila y ejecuta en SLURM)
- `prueba.sh` (envia casos de prueba)
- En caso de falla por fin de línea emplear:

```
 sed -i 's/\r$//' *
```

## Ejecutar todos los casos de prueba

```bash
bash prueba.sh
```

## Visualizar salidas nuevamente

```bash
cat salidas.sh
```

**Aclaración:**
Se formatea la salida del reporte nsys, para que solo muestre las métricas del kernel
