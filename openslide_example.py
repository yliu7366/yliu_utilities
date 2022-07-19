INPUTS = glob.glob(os.path.join(INPUT, '*.svs'))

for w in INPUTS:
  wsi = openslide.OpenSlide(w)
  wsiNpy = np.array(wsi.read_region((0,0), 0, (wsi.dimensions[0], wsi.dimensions[1])))[:,:,:3]
