The dataset includes 6391 images.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Randomly crop between 0 and 20 percent of the image
* Random rotation of between -15 and +15 degrees
* Random brightness adjustment of between -15 and +15 percent
* Random Gaussian blur of between 0 and 2.5 pixels
* Salt and pepper noise was applied to 6.36 percent of pixels


