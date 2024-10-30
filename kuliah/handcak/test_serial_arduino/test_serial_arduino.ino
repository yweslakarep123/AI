const int enA = 3;
const int in1 = 2;
const int in2 = 4;

String x;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(1);
  pinMode(LED_BUILTIN, OUTPUT);

  pinMode(enA, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
}

void loop() {
  while (!Serial.available());
  x = Serial.readString();
  Serial.print(x);
  if (x == "1"){
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    analogWrite(enA, 51);
    // digitalWrite(LED_BUILTIN, HIGH);
    // delay(3000);
    // digitalWrite(LED_BUILTIN, LOW);
    // delay(3000);
  }
  else if (x == "2"){
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    analogWrite(enA, 102);
    // digitalWrite(LED_BUILTIN, HIGH);
    // delay(1000);
    // digitalWrite(LED_BUILTIN, LOW);
    // delay(1000);
  }
  else if (x == "3"){
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    analogWrite(enA, 153);
  }
  else if (x == "4"){
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    analogWrite(enA, 204);
  }
  else if (x == "5"){
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    analogWrite(enA, 255);
  }
}