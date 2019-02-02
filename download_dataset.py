import urllib.request
import os
import zipfile

url = "https://www.cmlab.csie.ntu.edu.tw/project/Deep-Demosaic/static/dataset.zip"
zip_path = "data/dataset.zip"

# Download dataset .zip
print("Downloading dataset...")
response = urllib.request.urlretrieve(url, zip_path)

print("Unzipping...")
# Unzip
zip_ref = zipfile.ZipFile(zip_path, 'r')
zip_ref.extractall("data/")
zip_ref.close()

# Split up training and validation data
# Img491 through Img500 will be moved to validation data
print("Moving images to validation set:")
for i in range(491, 501):
    filename = ("Img%03d" % i) + ".png"
    print("\t" + filename)
    os.rename("data/Flickr500/" + filename,
              "data/Validation/" + filename)

print("Done!")
