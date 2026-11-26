from board_link import Board

board = Board("COM5", baud=921600)
print(board.info())