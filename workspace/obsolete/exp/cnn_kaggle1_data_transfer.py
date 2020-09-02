from root.utils import * # pylint: disable= unused-wildcard-import
from root.readData import readData, readCSV

from fastai.vision import * # pylint: disable= unused-wildcard-import
from fastai.metrics import error_rate

#train_rgb_images, df, train_labels = readData(100)


df = readCSV('train')


#path  = Path(os.path.realpath(__file__)[0:-14] + '/dataset/jpeg/')
path  = Path('dataset/jpeg/')
bs = 64


df = df[['img_name', 'target']]
df.columns = ['name','label']

data = pd.DataFrame(df[df['label']==1])
data2 = pd.DataFrame(df[df['label']==0])
data2 = data2[:584]
data = data.append(data2)
df2 = data


tfms = get_transforms(flip_vert=True)
src = (ImageList.from_df(df2, path, suffix ='.jpg', folder = 'train')
    .split_by_rand_pct(0.2)
    .label_from_df()
    )
data = (src.transform(tfms, size =256).databunch(bs=bs).normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(7,6))



learn = cnn_learner(data, models.resnet50, metrics=[error_rate,accuracy])
print(learn.model)



