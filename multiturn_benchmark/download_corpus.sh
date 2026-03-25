#!/bin/bash
set -euo pipefail

# Large Project Gutenberg books (English, plain text)
BOOKS=(
    "2600"    # War and Peace - Tolstoy
    "1342"    # Pride and Prejudice - Austen  
    "84"      # Frankenstein - Shelley
    "1661"    # Sherlock Holmes - Doyle
    "11"      # Alice in Wonderland - Carroll
    "1952"    # The Yellow Wallpaper - Gilman
    "98"      # A Tale of Two Cities - Dickens
    "1400"    # Great Expectations - Dickens
    "16328"   # Beowulf
    "2701"    # Moby Dick - Melville
    "4300"    # Ulysses - Joyce
    "5200"    # Metamorphosis - Kafka
    "1260"    # Jane Eyre - Bronte
    "768"     # Wuthering Heights - Bronte
    "74"      # Tom Sawyer - Twain
    "76"      # Huckleberry Finn - Twain
    "46"      # A Christmas Carol - Dickens
    "1080"    # A Modest Proposal - Swift
    "345"     # Dracula - Stoker
    "43"      # The Strange Case of Dr Jekyll and Mr Hyde
    "174"     # The Picture of Dorian Gray - Wilde
    "1232"    # The Prince - Machiavelli
    "25344"   # The Scarlet Letter - Hawthorne
    "2542"    # A Doll's House - Ibsen
    "160"     # The Awakening - Chopin
    "3207"    # Leviathan - Hobbes
    "996"     # Don Quixote - Cervantes
    "2554"    # Crime and Punishment - Dostoevsky
    "28054"   # The Brothers Karamazov - Dostoevsky
    "7178"    # Les Misérables - Hugo (English)
    "135"     # Les Misérables - Hugo
    "1497"    # Republic - Plato
    "5827"    # The Problems of Philosophy - Russell
    "730"     # Oliver Twist - Dickens
    "766"     # David Copperfield - Dickens
    "1023"    # Bleak House - Dickens
    "580"     # The Pickwick Papers - Dickens
    "883"     # The Origin of Species - Darwin
    "36"      # The War of the Worlds - Wells
    "35"      # The Time Machine - Wells
    "215"     # The Call of the Wild - London
    "1727"    # The Odyssey - Homer
    "6130"    # The Iliad - Homer
    "3600"    # The History of the Decline and Fall of the Roman Empire v1
    "3608"    # ...v2
    "3609"    # ...v3
)

OUTPUT="gutenberg_11m.txt"
rm -f "$OUTPUT"

echo "Downloading ${#BOOKS[@]} books from Project Gutenberg..."
for id in "${BOOKS[@]}"; do
    url="https://www.gutenberg.org/files/${id}/${id}-0.txt"
    echo -n "  PG#${id}... "
    if curl -sf "$url" >> "$OUTPUT" 2>/dev/null; then
        echo "ok"
    else
        # Try alternate URL format
        url="https://www.gutenberg.org/cache/epub/${id}/pg${id}.txt"
        if curl -sf "$url" >> "$OUTPUT" 2>/dev/null; then
            echo "ok (alt)"
        else
            echo "skip"
        fi
    fi
    echo "" >> "$OUTPUT"  # separator between books
done

SIZE=$(wc -c < "$OUTPUT")
echo "Done: $OUTPUT ($(numfmt --to=iec $SIZE))"
