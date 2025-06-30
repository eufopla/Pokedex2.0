!!! This small pokedex doesn't work !!!

I made this project to understand and create a neural network that "learns" et recognize 4 types of Pokemons.
Pikachu, Elektek, Lucario, and Cosmog. (The names are in french)

The datasets don't contains enough images in order to make the AI recognize one of these Pokemons.
But, the process is functionnal and we can see the unfolding of how the AI learns, the number of epochs, the differents percentage.

I added the early stop because in the way that I made it, there to less data and to many epochs. 
This is called overfitting, it learns to many details and cannot recognize a Pokemon simply. 
So when it reaches 100% of accuracy, it tolerates 4 more epochs where there's no more amelioration and stops by itself.

!!! In order to make it "work" !!!

You need an virtual environnement of python in 3.11, activate it (venv\Scripts\activate.bat)
And in it, you need to install tensorflow 2.15 
I used vscode, so I needed to install every libraries with pip install.

In the source code, you need to change the path of the image to predict according to wich image your want to try.
