from fastapi import FastAPI, Request, Form

from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles # pour css et images

from fastapi import Request

app = FastAPI()

app.mount(
    "/static",
    StaticFiles(directory= "static"),
    name="static",
)

# definition templates
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def read_root(request: Request):
    # index.html dans template
    return templates.TemplateResponse("index.html", {"request": request, "message": None})

# webcam.html dans template
@app.get("/webcam")
async def afficher_page_webcam(request: Request):
    return templates.TemplateResponse(name="webcam.html", context={"request": request})

@app.post("/record")
async def record(request: Request):

    form_data = await request.form()  # Parse the form data
    form_dict = dict(form_data)  # Convert to a standard dictionary
    print( {"received_data": form_dict})
    # Process the data on the server side
    
    #print(start_time)
    texte = form_dict["texte"]
    start = form_dict["start_time"]
    response_message = f"Received data: {texte} {start}"
    print(response_message)
    # renvoie des data apres traitement au format json
    return JSONResponse(content={"message": response_message})
    #sert si on fait que de l'html
    #return templates.TemplateResponse("index.html", {"request": request, "message": response_message})
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8010)