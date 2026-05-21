import 'react-native-gesture-handler';
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { StatusBar } from 'expo-status-bar';

import FeedScreen from './screens/FeedScreen';
import AddReelScreen from './screens/AddReelScreen';
import { COLORS } from './constants';

const Stack = createStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <StatusBar style="light" />
      <Stack.Navigator
        initialRouteName="Feed"
        screenOptions={{
          headerShown: false,        // custom top bars in each screen
          cardStyle: { backgroundColor: COLORS.bg },
          animation: 'slide_from_right',
        }}
      >
        <Stack.Screen name="Feed" component={FeedScreen} />
        <Stack.Screen name="AddReel" component={AddReelScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
