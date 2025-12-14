import requests

def probar_api():

    base_url = "http://localhost:8000"

    print("verificando health...")
    r = requests.get(f"{base_url}/health")
    print(r.json())

    print("probando prediccion...")

    datos_cliente = {
        "tenure": 5,
        "MonthlyCharges": 85.5,
        "TotalCharges": 420.0,
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "InternetService": "Fiber optic"
    }

    r = requests.post(f"{base_url}/predict", json=datos_cliente)
    print(r.json())


if __name__ == "__main__":
    probar_api()
