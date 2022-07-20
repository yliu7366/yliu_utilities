import PIL
from PIL import Image

#bypass the size limit
Image.MAX_IMAGE_PIXELS = 200000*200000

INPUTS = glob.glob(os.path.join(INPUT, '*.svs'))

for w in INPUTS:
  wsi = openslide.OpenSlide(w)
  wsiNpy = np.array(wsi.read_region((0,0), 0, (wsi.dimensions[0], wsi.dimensions[1])))[:,:,:3]

#take out a 256x256 patch from top left (0, 0)
roi = wsiNpy[0:256, 0:256]
#take out a 256x256 patch from (256, 0)
roi = wsiNpy[0:256, 256:512]
