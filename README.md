# Overcooked Demo
A web application where humans can play Overcooked with trained AI agents.

## Installation

First, install `node` and `npm`, and ensure that you have set up the [overcooked-js repository](https://github.com/markkho/overcooked-js).

Then, at the top-level directory:

    overcooked-demo $ npm install
    overcooked-demo $ npm link /path/to/overcooked-js/
    overcooked-demo $ npm run build

At this point the code should be ready to run. Start the server with

    overcooked-demo $ node index.js

at which point you should be able to load the website in any browser by going to [http://localhost:8766](http://localhost:8766).


If you want to build the trajectory replay code, all of the above is the same, 
but you'll need to run `npm run build-replay`, and then navigate to [http://localhost:8766/replay](http://localhost:8766/replay).
