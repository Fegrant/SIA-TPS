# SIA-TPS

# Para correrlo

pipenv install
pipenv shell

### Random grid 
python solver.py -r size color_amount max_turns
Por ejemplo python solver.py -r 3 5 30

### With input file
python solver.py file_path max_turns
Por ejemplo python solver ../input_grid/3x3.txt 30