Conference_Pairings
===================

Copyright (C)
Federico Vaggi
Tommaso Schiavinotto
Attila Csikasz-Nagy
Rafael Carazo-Salas

Distributed under AGPL3; see LICENSE.txt.  For use of the code under a
different license, or information about consulting and solutions tailored
to specific buisness needs, please visit: http://www.matchingknowledge.com/


Summary:
--------
Code used to implement the matchmaking procedure described in Vaggi et al,
eLife, 2014.

Given a list of conference participants and their connections, the methods
that they know, the methods that they wish to know, speed_dating.py creates
multiple unique rounds that match the participants according to user specified
criteria.

By default, two different sets of 5 'dates' are created.  The first set is
between people who are mutually interested in each other's methods.  The
second set is between people who know different methods.  Details of the
method are described in the publication.

Requirements:
-------------
- SciPy
- NumPy
- NetworkX
- Pandas

Input Files:
------------
Example of inputs are available in the Input directory.
- Network.txt: A file, in edgelist format, containing all the connections
between the participants of the meeting.
- Known_Methods: A .csv file containing the known methods by all participants.
- Wanted_Methods: A .csv file containing the wanted methods by all
participants.

Output:
-------
By default, the output is directed to standard output.  It's possible
to specify a desired output file in Settings.json file.
