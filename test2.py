from text import symbols, text_to_sequence, sequence_to_text

t = 'phải{sp}zcó'
x = text_to_sequence(t, [])
print(x, len(x))
print(sequence_to_text(x))