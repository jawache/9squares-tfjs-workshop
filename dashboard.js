const SQ_SIZE = 5;

const BACKGROUND_COLOR = 50;
const TEXT_COLOR = 255;

class Dashboard {
  constructor() {
    this.data = null;
    this.squares = new Squares();
    this.details = new Details();
  }

  setData(d) {
    this.data = d;
    this.squares.setData(d);
    this.details.setData(d);
  }

  getData() {
    return this.data;
  }

  draw() {
    // Add Title
    background(BACKGROUND_COLOR);
    translate(10, 10);
    fill(TEXT_COLOR)
      .strokeWeight(0)
      .textFont("Helvetica", 24);
    text("9 Squares", 0, 24);
    translate(0, 40);

    if (this.data) {
      // Add Squares
      this.squares.draw();

      translate(450, 0);
      this.details.draw();
    }
  }
}

// class Panel {
//   constructor(title, x, y, width, height) {
//     this.title = title;
//     this.x = x;
//     this.y = y;
//     // this.width = width;
//     // this.height = height;
//     this.children = [];
//   }

//   add(child) {
//     this.children.push(child);
//     return this;
//   }

//   draw() {
//     push();
//     // Position this panel where it's designed to be
//     translate(this.x, this.y);

//     // Type the title
//     // fill(255)
//     //   .strokeWeight(0)
//     //   .textSize(16)
//     //   .textFont("Helvetica", 12);
//     // text(this.title, 0, 15);
//     // Move down a bit
//     translate(0, 20);

//     for (let child of this.children) {
//       child.draw(this.width, this.height);
//     }
//     pop();
//   }
// }

class Squares {
  constructor(data) {
    this.data = data;
    this.squares = [];
  }

  setData({
    inputs,
    labels,
    predictions,
    loss,
    weights,
    testingLoss,
    testingAcc
  }) {
    this.loss = loss;
    this.weights = weights;
    this.testingLoss = testingLoss;
    this.testingAcc = testingAcc;
    this.squares = [];
    for (let i = 0; i < inputs.length; i++) {
      let input = inputs[i].map(x => Math.round(x * 255));
      let label = labels[i];
      let predicted = predictions ? predictions[i] : "";
      this.squares.push(new Square(i, input, label, predicted));
    }
  }

  draw(width, height) {
    // Draw each individual square
    push();

    for (let i = 0; i < this.squares.length; i++) {
      push();
      // Show 25 per row
      const col = i % 25;
      const row = parseInt(Math.floor(i / 25));

      // Each is 15 width and 25 height, add 2 px padding per
      const x = col * 17;
      const y = row * 27;
      translate(x, y);

      this.squares[i].draw(width, height);

      pop();
    }
    pop();
  }
}

class Square {
  constructor(id, inputs, label, predicted) {
    this.id = id;
    this.inputs = inputs;
    this.label = label;
    this.predicted = predicted;
  }
  draw() {
    push();
    noStroke();

    rect(0, 0, SQ_SIZE, SQ_SIZE);
    fill(this.inputs[0]);
    rect(5, 0, SQ_SIZE, SQ_SIZE);
    fill(this.inputs[1]);
    rect(10, 0, SQ_SIZE, SQ_SIZE);
    fill(this.inputs[2]);
    rect(0, 5, SQ_SIZE, SQ_SIZE);
    fill(this.inputs[3]);
    rect(5, 5, SQ_SIZE, SQ_SIZE);
    fill(this.inputs[4]);
    rect(10, 5, SQ_SIZE, SQ_SIZE);
    fill(this.inputs[5]);
    rect(0, 10, SQ_SIZE, SQ_SIZE);
    fill(this.inputs[6]);
    rect(5, 10, SQ_SIZE, SQ_SIZE);
    fill(this.inputs[7]);
    rect(10, 10, SQ_SIZE, SQ_SIZE);
    fill(this.inputs[0]);

    // Labels & Predictions
    let color = this.label == this.predicted ? "green" : "red";
    // const icon = x => (x === 1 ? "▁" : "▔");
    const icon = x => (x === 1 ? "▁" : "▔");

    textSize(6);
    fill(color);
    rect(0, 15, 7.5, 10);
    rect(7.5, 15, 7.5, 10);

    fill("white");
    text(icon(this.label), 1, 22.5);
    text(icon(this.predicted), 7, 22.5);

    pop();
  }
}

class Details {
  constructor() {
    this.counter = 0;
  }

  setData(d) {
    this.data = d;
  }

  draw() {
    const { loss, weights, testingLoss, testingAcc, epoch } = this.data;
    this.counter++;

    // Type the title
    fill(255)
      .strokeWeight(0)
      .textFont("Helvetica", 12);

    text(`EPOCH:`, 0, 20);
    text(`LOSS:`, 0, 40);
    text(`ACCURACY:`, 0, 60);
    text(`WEIGHTS:`, 0, 80);

    textFont("Courier", 12);

    text(epoch, 100, 20);
    text(loss.toFixed(4), 100, 40);
    text(`${(testingAcc * 100).toFixed(2)}%`, 100, 60);
    push();
    translate(100, 120);
    for (let i = 0; i < 9; i++) {
      const col = i % 3;
      const row = parseInt(Math.floor(i / 3));
      const w = weights[i].toFixed(5);
      const x = col * 80;
      const y = row * 20;
      text(w, x, y);
    }
    pop();

    push();
    {
      let input = this.data.inputs[
        parseInt(
          Math.floor(Math.abs(Math.sin(epoch) * this.data.inputs.length))
        )
      ].map(x => Math.round(x * 255));
      const size = 25;
      translate(0, 100);
      strokeWeight(2);
      stroke(255);
      noFill();
      rect(0, 0, size, size);
      fill(input[0]);
      rect(size, 0, size, size);
      fill(input[1]);
      rect(2 * size, 0, size, size);
      fill(input[2]);
      rect(0, size, size, size);
      fill(input[3]);
      rect(size, size, size, size);
      fill(input[4]);
      rect(2 * size, size, size, size);
      fill(input[5]);
      rect(0, 2 * size, size, size);
      fill(input[6]);
      rect(size, 2 * size, size, size);
      fill(input[7]);
      rect(2 * size, 2 * size, size, size);
      fill(input[8]);
    }
    pop();
  }
}
