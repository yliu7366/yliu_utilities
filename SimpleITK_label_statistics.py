for image in files:
        img = sitk.ReadImage(os.path.join(originalPath, image))
        imgPredict = sitk.ReadImage(os.path.join(labelPath, os.path.splitext(image)[0] + '.png'))

        labels = sitk.RelabelComponent(sitk.ConnectedComponent(imgPredict))
        labelsNpy = sitk.GetArrayFromImage(labels)
        print(labelsNpy.shape, np.max(labelsNpy))
        np.save(os.path.join(relabelPath, os.path.splitext(image)[0] + '.npy'), labelsNpy)

        shapeLabelFilter = sitk.LabelShapeStatisticsImageFilter()
        shapeLabelFilter.Execute(labels)

        csvFile = os.path.splitext(image)[0] + '.csv'
        with open(os.path.join(csvPath, csvFile), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id', 'roundness', 'pixels', 'bboxX', 'bboxY', 'bboxWidth', 'bboxHeight'])
            for i in range(1, shapeLabelFilter.GetNumberOfLabels()+1):
                writer.writerow([i, shapeLabelFilter.GetRoundness(i),
                                shapeLabelFilter.GetNumberOfPixels(i), 
                                shapeLabelFilter.GetBoundingBox(i)[0],
                                shapeLabelFilter.GetBoundingBox(i)[1],
                                shapeLabelFilter.GetBoundingBox(i)[2],
                                shapeLabelFilter.GetBoundingBox(i)[3]])
        print(os.path.join(csvPath, csvFile))
