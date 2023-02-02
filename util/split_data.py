# Do first time before extract zip
# If "Immunofluorescence images" folder is already split. Don't run this code
import os
from tqdm import tqdm

dapi = []
fitc = []
merge = []

for file in tqdm(sorted(os.listdir('Immunofluorescence images'))):
    if file.endswith('.tif.tif'):
        os.rename('Immunofluorescence images/' + file, 'Immunofluorescence images/' + file[:-4])
        file = file[:-4]

    if 'DAPI' in file:
        dapi.append(file)
    elif 'FITC' in file:
        fitc.append(file)
    elif 'merge' in file:
        merge.append(file)


print('DAPI: ', len(dapi))
print('FITC: ', len(fitc))
print('Merge: ', len(merge))

os.mkdir('Immunofluorescence images/DAPI')
os.mkdir('Immunofluorescence images/FITC')
os.mkdir('Immunofluorescence images/Merge')

for file in tqdm(dapi):
    os.rename('Immunofluorescence images/' + file, 'Immunofluorescence images/DAPI/' + file)

for file in tqdm(fitc):
    os.rename('Immunofluorescence images/' + file, 'Immunofluorescence images/FITC/' + file)

for file in tqdm(merge):
    os.rename('Immunofluorescence images/' + file, 'Immunofluorescence images/Merge/' + file)

print('Complete moving files')