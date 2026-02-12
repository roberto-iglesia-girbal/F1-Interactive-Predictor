# F1 Simulator 2026

Simulador de carreras de Fórmula 1 utilizando Machine Learning para predecir resultados y estrategias, basado en datos históricos y telemetría en tiempo real.

### Demo

![Gif Simulador F1 2026](f1-simulator-2026.gif)



### Instalación

1. Clona este repositorio o descarga los archivos.
2. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```

### Uso

- Windows
Simplemente ejecuta el script automático:
```cmd
run_simulation.bat
```

- Linux / Mac
Dale permisos de ejecución y corre el script:
```bash
chmod +x startup.sh
./startup.sh
```

### Manual
Si prefieres ejecutarlo manualmente:
1. Ejecuta el servidor backend desde la raíz del proyecto:
   ```bash
   python backend/f1-predictor.py
   ```
2. El servidor iniciará en el puerto 5050.
3. Abre tu navegador web y visita: `http://localhost:5050`


