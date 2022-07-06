# Project

Welcome! This is the implementation of the manga paper by group IFHTP: 

http://www.cse.cuhk.edu.hk/~ttwong/papers/manga/manga.pdf



#### Dependencies

Make sure that you have latest version of pip and python3.7 installed.
Here are some of the dependencies listed:

* numpy 
* matplotlib.pyplot 
* opencv-contrib-python
* cv2
* webcolors 
* scikit-image
* scipy 

How to install these dependencies: 
```
pip3 install <package name>
```

---


To run the program, run:
```bash
python main.py <image> <config-file>
```
The config files are present in ./configs directory.
You can find the reference config in config.json.
Make sure that code.py should be in the same directory as main.py

Interactive Manga Colorization.
GUI Usage: 
On running the program, three windows show up, 
Input window shows the input image and the user scribble are done here
Output window accepts the keyboard inputs and displays the final image
Trackbar window changes the color of the scribble

Select the output window and press '0' for intensity continuous
segmentation and '1' for pattern continuous segmentation
The program shows the boundaries obtained after every 10 iterations
Pressing e here terminates the further executions and end the segmentation procedure
You can also press ctrl-c for ending the segmentation at any moment

After the segments are obtained, select the output window  and press 0 for 
color replacement colorization, 1 for stroke preserving colorization, and 2 for
pattern to shading colorization


Link to input directory: [input_dir](https://drive.google.com/drive/folders/1o9GbP97ypw0nbnKsTo3EyOszMw1Qzx76?usp=sharing)

Link to output directory: [output_dir](https://drive.google.com/drive/folders/1N4laS-EKntXNhJC2gEgNyHF1gJqxH2DA?usp=sharing)



















