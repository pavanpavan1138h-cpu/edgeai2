from board_link import Board

board = Board("COM5", baud=921600)

try:
    print("Flash CRC:", board.crc("flash"))
    flash_data = board.dump("flash")
    print("Flash bytes after restart:", len(flash_data))
finally:
    board.close()