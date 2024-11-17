FROM node:18

WORKDIR /app

COPY package*.json ./

RUN rm -rf node_modules package-lock.json
RUN npm cache clean --force
RUN npm install

RUN npm install -g react-scripts

COPY . .

RUN npm install

EXPOSE 3000

CMD ["npx", "react-scripts", "start"]