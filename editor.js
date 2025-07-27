Office.onReady(() => {
  document.getElementById("checkButton").onclick = () => {
    Office.context.mailbox.item.body.getAsync("text", {}, function (result) {
      if (result.status === Office.AsyncResultStatus.Succeeded) {
        const emailText = result.value;
        fetch("http://localhost:8000/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: emailText }),
        })
          .then((response) => response.json())
          .then((data) => {
            document.getElementById("result").innerText =
              `Confidence: ${data.confidence}%\nLabel: ${data.label}`;
          })
          .catch((error) => {
            console.error(error);
            document.getElementById("result").innerText = "API call error";
          });
      } else {
        document.getElementById("result").innerText = "Failed to get email body";
      }
    });
  };
});