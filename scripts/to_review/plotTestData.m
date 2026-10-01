clear
load("test.mat")

innerRows = strcmp(testLabels, "InnerRaceFault");
innerData = testData(innerRows, :);

figure

subplot(2,1,1)
plot(innerData(1,:))
title("First InnerRaceFault Recording")
xlabel("Sample number")
ylabel("Signal value")
grid on

subplot(2,1,2)
plot(innerData(end,:))
title("Last InnerRaceFault Recording")
xlabel("Sample number")
ylabel("Signal value")
grid on