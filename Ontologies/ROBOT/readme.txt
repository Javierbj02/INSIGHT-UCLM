First approach using the ROBOT tool in order to create ontologies from reference ontologies.

To install ROBOT check https://robot.obolibrary.org/

In /src:

robot extract --input DUL.owl.xml --method BOT --term-file terms.txt --output DUL.subset.owl