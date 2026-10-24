# QuantMove

Motor cuantitativo de valoración y gestión de riesgos de portafolios.

## Funcionalidades

- Validación de datos de precios.
- Descarga de precios ajustados con yfinance.
- Estadísticas de rendimientos, correlación y volatilidad.
- Estimación histórica de VaR y Expected Shortfall (ES).
- Simulación de riesgo mediante Monte Carlo.
- Comparación de estimadores de riesgo.
- Backtesting y pruebas estadísticas de cobertura.
- Suite de pruebas automatizadas con pytest.

## Requisitos

Python y las dependencias indicadas en los archivos de requisitos.
Se necesita internet para descargar datos de mercado.

## Instalación

Desde la raíz del proyecto, ejecuta:

```bash
pip install -r requirements-dev.txt
```

## Ejecutar las pruebas

```bash
python -m pytest -q
```

## Ejemplo de uso

```python
from quantmove.data.market_data import download_adjusted_close
from quantmove.portfolio.historical_risk_pipeline import run_historical_risk_pipeline

prices = download_adjusted_close(tickers=["AAPL", "MSFT"], period="1y", interval="1d")
weights = {"AAPL": 0.50, "MSFT": 0.50}

result = run_historical_risk_pipeline(
    prices=prices,
    weights=weights,
    confidence=0.95,
    window_size=60,
    significance_level=0.05,
    n_simulations=10_000,
    random_state=42,
)

print(result["historical_var_es"])
print(result["backtest_report"])
```

El ejemplo requiere conexión a internet. Los resultados dependen
de los datos obtenidos y de los parámetros utilizados.

## Estructura del proyecto

- `configs/`: configuraciones.
- `docs/`: documentación.
- `scripts/`: scripts auxiliares.
- `src/quantmove/`: código fuente.
- `tests/`: pruebas automatizadas.
- `pytest.ini`: configuración de pytest.
- `requirements.txt`: dependencias de ejecución.
- `requirements-dev.txt`: dependencias de desarrollo y pruebas.

## Estado del proyecto

En desarrollo. Las interfaces y la documentación pueden cambiar.

## Aviso

QuantMove es una herramienta de análisis cuantitativo en desarrollo.
No constituye asesoramiento financiero ni garantiza resultados futuros.
