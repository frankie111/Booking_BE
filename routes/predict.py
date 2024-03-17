# @app.post("/predict/")
# def predict_availability(request: PredictionRequest):
#     try:
#         input_data = prepare_input_data(request.date, request.room_name)
#         # Reshape input for a single sample
#         prediction = model.predict([input_data])
#         availability = "Available" if prediction[0] == 1 else "Not Available"
#         return {"date": request.date.strftime("%Y-%m-%d"), "room_name": request.room_name, "availability": availability}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))