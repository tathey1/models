The scripts convert_tiff_to_jpeg.py shows how I converted the tif images to jpegs (which was necessary in order to make tfrecord files)
The file datasets/download_and_convert_pathology_xval.py shows how I made the 10 splits and created the tfrecord files

After modifying xval.sh, you can run it with ./xval.sh to perform 10 fold cross validation on a model

I will outline the changes that need to be made to create a new model that is based on pretrained weights
In this example, say your net is based on resnet:

1) create the net within the file nets/resnet_v1.py
	a) create the method that outlines the net, based on resnet_v1, such as resnet_v1_final
		this outline may incorporate external methods such as _tile_images, _max_tile
	b) create the method that describes the specifics of the net, based on resnet_v1_50, such as resnet_v1_50_final
2) in nets/net_factory.py
	a) add the net in networks_map
	b) add an entry in the argscope dictionary, I just copy resnet_v1.resnet_arg_scope
3) If you want, you can make a custom preprocessing script based on preprocess/vgg_preprocess
4) Add the preprocessing function in preprocess/preprocess_factory.py preprocess_fn_map
5) Assuming that you made a new net named, e.g. 'resnet_v1_50_test' that borrows weights from 'resnet_v1_50', specify the mapping in checkpoints/checkpoints_map.py
	a) in line 21, there is a string replace command, this deals with the scopes that were specified back in step 1
	b) e.g. if all of my new scopes start with resnet_v1_50_test, I want to remove the'_test' substring to make sure there is a correct mapping of pretrained weights from resnet_v1_50 to resnet_v1_50_test
