from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai.vision import *

model_file_url = 'https://drive.google.com/uc?export=download&id=16cm9MnWM18myoCDeLupJSrvRGjSENsfC'
model_file_name = 'model'
classes = ['Apple Braeburn',
 'Apple Crimson Snow',
 'Apple Golden 1',
 'Apple Golden 2',
 'Apple Golden 3',
 'Apple Granny Smith',
 'Apple Pink Lady',
 'Apple Red 1',
 'Apple Red 2',
 'Apple Red 3',
 'Apple Red Delicious',
 'Apple Red Yellow 1',
 'Apple Red Yellow 2',
 'Apricot',
 'Avocado',
 'Avocado ripe',
 'Banana',
 'Banana Lady Finger',
 'Banana Red',
 'Beetroot',
 'Blueberry',
 'Cactus fruit',
 'Cantaloupe 1',
 'Cantaloupe 2',
 'Carambula',
 'Cauliflower',
 'Cherry 1',
 'Cherry 2',
 'Cherry Rainier',
 'Cherry Wax Black',
 'Cherry Wax Red',
 'Cherry Wax Yellow',
 'Chestnut',
 'Clementine',
 'Cocos',
 'Dates',
 'Eggplant',
 'Ginger Root',
 'Granadilla',
 'Grape Blue',
 'Grape Pink',
 'Grape White',
 'Grape White 2',
 'Grape White 3',
 'Grape White 4',
 'Grapefruit Pink',
 'Grapefruit White',
 'Guava',
 'Hazelnut',
 'Huckleberry',
 'Kaki',
 'Kiwi',
 'Kohlrabi',
 'Kumquats',
 'Lemon',
 'Lemon Meyer',
 'Limes',
 'Lychee',
 'Mandarine',
 'Mango',
 'Mango Red',
 'Mangostan',
 'Maracuja',
 'Melon Piel de Sapo',
 'Mulberry',
 'Nectarine',
 'Nectarine Flat',
 'Nut Forest',
 'Nut Pecan',
 'Onion Red',
 'Onion Red Peeled',
 'Onion White',
 'Orange',
 'Papaya',
 'Passion Fruit',
 'Peach',
 'Peach 2',
 'Peach Flat',
 'Pear',
 'Pear Abate',
 'Pear Forelle',
 'Pear Kaiser',
 'Pear Monster',
 'Pear Red',
 'Pear Williams',
 'Pepino',
 'Pepper Green',
 'Pepper Red',
 'Pepper Yellow',
 'Physalis',
 'Physalis with Husk',
 'Pineapple',
 'Pineapple Mini',
 'Pitahaya Red',
 'Plum',
 'Plum 2',
 'Plum 3',
 'Pomegranate',
 'Pomelo Sweetie',
 'Potato Red',
 'Potato Red Washed',
 'Potato Sweet',
 'Potato White',
 'Quince',
 'Rambutan',
 'Raspberry',
 'Redcurrant',
 'Salak',
 'Strawberry',
 'Strawberry Wedge',
 'Tamarillo',
 'Tangelo',
 'Tomato 1',
 'Tomato 2',
 'Tomato 3',
 'Tomato 4',
 'Tomato Cherry Red',
 'Tomato Maroon',
 'Tomato Yellow',
 'Walnut']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')

    data = (ImageList.from_folder(path_train) #Where to find the data? -> in path and its subfolders
    .split_by_rand_pct()              #How to split in train/valid? -> use the folders
    .label_from_folder()            #How to label? -> depending on the folder of the filenames
    .add_test_folder(test_folder='/root/.fastai/data/Fruit-Images-Dataset-master/Test/')   #Optionally add a test set (here default name is test)
    .transform(tfms, size=64)       #Data augmentation? -> use tfms with a size of 64
    .databunch())   
    learn = cnn_learner(data, models.resnet34, metrics=accuracy)
    learn.load(model_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    return JSONResponse({'result': learn.predict(img)[0]})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=8080)

