
// This is the code behind this visualization which shows the internal vs external validity gained by having y number of transactions with x number of control stores assuming 1000 stores in the population.

const calculateValidity = (transactions, stores) => {
  // Internval vs External Validity Calculation
  const internalValidity = Math.min(100, (transactions / 50) * 100); // 50 transactions as "ideal"
  const externalValidity = Math.min(100, (stores / 100) * 100); // 100 stores as "ideal"
  
  // Simulate partial pooling effect 
  // (i.e. some stores will have fewer and some stores will have more transactions - partial pooling helps us share information across stores)
  const poolingEffect = Math.min(20, (stores / 10) * (100 - internalValidity) / 100);
  const adjustedInternalValidity = Math.min(100, internalValidity + poolingEffect);
  
  return { internalValidity: adjustedInternalValidity, externalValidity };
};

const generateData = () => {
  const data = [];
  for (let stores = 10; stores <= 200; stores += 10) {
    for (let transactions = 5; transactions <= 100; transactions += 5) {
      const { internalValidity, externalValidity } = calculateValidity(transactions, stores);
      data.push({ stores, transactions, internalValidity, externalValidity });
    }
  }
  return data;
};

const data = generateData();

const getColor = (internal, external) => {
  if (internal >= 80 && external >= 80) return "#4CAF50";  // High both: Green
  if (internal >= 80) return "#2196F3";  // High internal only: Blue
  if (external >= 80) return "#FFC107";  // High external only: Yellow
  return "#F44336";  // Low both: Red
};
