#modified based on recorded ImageJ Marco to perform color deconvolution on all images 

run("Close All");

dir = "./input";
outputDir = "./output"

files = getFileList(dir)

for(i=0; i<files.length; i++)
{
	open(dir+files[i]);
	run("Colour Deconvolution", "vectors=[H&E DAB] hide");
	selectWindow(files[i]+"-(Colour_1)");
	saveAs("PNG", outputDir+files[i]+"-c1.png");
	selectWindow(files[i]+"-(Colour_2)");
	saveAs("PNG", outputDir+files[i]+"-c2.png");
	selectWindow(files[i]+"-(Colour_3)");
	saveAs("PNG", outputDir+files[i]+"-c3.png");
	run("Close All");
}
